"""Decode the Tuya ``m/m/i/<devId>`` live-map stream used by Tuya-cloud models (L60 SES).

These devices never publish on the eufy biz/ topic. The app instead subscribes to
``m/m/i/<devId>`` on its Tuya mobile MQTT session and receives framed messages:

    55aa 0000 0002 | len u16 | 0001 | type u8 | sub u8 | u8 | ts u32 | seq u32 | plen u16 | payload | crc u32

Types (sub 0 carries data, sub 1 is a "renew" marker or heartbeat):
    2  map    -- delimited protobuf, f4 LZ4 block -> zlib -> 84 B header + u8 grid
    3  path   -- 5 B records: x, y u16 BE sign-magnitude [cm], flag u8; seq = byte offset
    4  rooms  -- delimited protobuf, f7 LZ4 block -> zlib -> per-room contour chains
    5  names  -- delimited protobuf, repeated f2 {1: id, 2: name}, same order as the chains
    7  pose   -- delimited protobuf {1: {1: x, 2: y, 3: theta}} sint32, cm / centiradians

Map, rooms and names only stream while the app has the map open; pose and path
always stream during a clean. The per-clean history file in Tuya storage embeds the
same map (f10), rooms (f11) and names (f3) payloads, so it is the app-free source.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from PIL import Image, ImageDraw

from ..utils import decode_varint
from .map_stream import MapData, _lz4_block_decompress

FRAME_MAP = 2
FRAME_PATH = 3
FRAME_ROOMS = 4
FRAME_ROOM_NAMES = 5
FRAME_POSE = 7

_FRAME_MAGIC = b"\x55\xaa"
_PAYLOAD_OFFSET = 23
_MAP_HEADER = struct.Struct("<15I5fI")

# Grid cell values -> map_stream 2-bit pixel values (0 unknown, 1 wall, 2 floor).
_CELL_TO_PIXEL = {127: 2, 128: 1}


@dataclass
class MmiFrame:
    """One framed message from the ``m/m/i`` topic."""

    kind: int
    sub: int
    ts: int
    seq: int
    payload: bytes


@dataclass
class RoomChain:
    """A room's ordered outline in map grid cells (row 0 = bottom)."""

    room_id: int
    cells: list[tuple[int, int]]


def parse_frame(raw: bytes) -> MmiFrame | None:
    """Split an ``m/m/i`` message into header fields and payload; ``None`` if not a frame."""
    if len(raw) < _PAYLOAD_OFFSET or raw[:2] != _FRAME_MAGIC:
        return None
    # Byte offsets follow the frame layout in the module docstring.
    (plen,) = struct.unpack_from(">H", raw, 21)
    ts, seq = struct.unpack_from(">II", raw, 13)
    return MmiFrame(raw[10], raw[11], ts, seq, raw[_PAYLOAD_OFFSET:_PAYLOAD_OFFSET + plen])


def _fields(buf: bytes) -> dict[int, list[int | bytes]]:
    """Walk a protobuf message, keeping varint and length-delimited fields only."""
    out: dict[int, list[int | bytes]] = {}
    pos = 0
    while pos < len(buf):
        tag, pos = decode_varint(buf, pos)
        wire = tag & 7
        value: int | bytes
        if wire == 0:
            value, pos = decode_varint(buf, pos)
        elif wire == 2:
            n, pos = decode_varint(buf, pos)
            value = buf[pos:pos + n]
            pos += n
        else:
            break
        out.setdefault(tag >> 3, []).append(value)
    return out


def _varint(fields: dict[int, list[int | bytes]], field_no: int) -> int:
    value = fields[field_no][0]
    if not isinstance(value, int):
        raise ValueError(f"protobuf field {field_no} is not a varint")
    return value


def _blobs(fields: dict[int, list[int | bytes]], field_no: int) -> list[bytes]:
    values = fields.get(field_no, [])
    blobs = [v for v in values if isinstance(v, bytes)]
    if len(blobs) != len(values):
        raise ValueError(f"protobuf field {field_no} is not length-delimited")
    return blobs


def undelimit(payload: bytes) -> bytes:
    """Strip the varint length prefix that live map/rooms/names/pose payloads carry."""
    n, pos = decode_varint(payload, 0)
    return payload[pos:pos + n]


def _unpack(fields: dict[int, list[int | bytes]], data_field: int) -> bytes:
    return zlib.decompress(_lz4_block_decompress(_blobs(fields, data_field)[0], _varint(fields, data_field + 1)))


def decode_map(message: bytes) -> MapData:
    """Decode a map protobuf (frame type 2 after un-delimiting, or history f10) into ``MapData``.

    The grid is re-packed into map_stream's 2-bit pixel format so ``render_map_png``
    can draw it; origin and resolution are converted to cm.
    """
    blob = _unpack(_fields(message), 4)
    header = _MAP_HEADER.unpack_from(blob)
    # Only these header fields are understood: u32 13-14 grid size, f32 15-16 origin [m],
    # f32 19 cell size [m]. The rest are left alone.
    width, height = header[13], header[14]
    min_x, min_y, resolution = header[15], header[16], header[19]
    cells = blob[_MAP_HEADER.size:_MAP_HEADER.size + width * height]
    packed = bytearray((width * height + 3) // 4)
    for i, cell in enumerate(cells):
        pixel = _CELL_TO_PIXEL.get(cell, 0)
        if pixel:
            packed[i >> 2] |= pixel << ((i & 3) * 2)
    return MapData(
        raw_pixels=bytes(packed),
        width=width,
        height=height,
        origin_x=round(min_x * 100),
        origin_y=round(min_y * 100),
        resolution=round(resolution * 100),
    )


def decode_rooms(message: bytes) -> list[RoomChain]:
    """Decode a rooms protobuf (frame type 4 after un-delimiting, or history f11) into outlines.

    Each contour point is (gx, gy, door) u32; door is a shared door-segment id and
    is not needed for filling, so it is dropped.
    """
    blob = _unpack(_fields(message), 7)
    # The leading u32 is not needed; the room count follows it, then per room
    # (id u32, n u32) and n contour points.
    (count,) = struct.unpack_from("<I", blob, 4)
    pos = 8
    chains = []
    for _ in range(count):
        room_id, n = struct.unpack_from("<II", blob, pos)
        pos += 8
        cells = [struct.unpack_from("<II", blob, pos + 12 * k) for k in range(n)]
        pos += 12 * n
        chains.append(RoomChain(room_id, cells))
    return chains


def decode_room_names(message: bytes) -> list[tuple[int, str]]:
    """Room ``(id, name)`` pairs from a names protobuf (frame type 5 after un-delimiting, or history f3).

    These ids are the ones ``room_clean`` takes. The outline chains use a separate
    numbering (chain id = room id + 9 on observed maps), but both lists come in the
    same order, so names are paired with chains by position.
    """
    rooms = []
    for entry in _blobs(_fields(message), 2):
        fields = _fields(entry)
        rooms.append((_varint(fields, 1), _blobs(fields, 2)[0].decode()))
    return rooms


def name_rooms(chains: list[RoomChain], names: list[tuple[int, str]]) -> dict[int, str]:
    """Map chain ids to names by position; empty if the two lists disagree in length."""
    if len(chains) != len(names):
        return {}
    return {chain.room_id: name for chain, (_room_id, name) in zip(chains, names)}


def apply_rooms(map_data: MapData, chains: list[RoomChain], names: dict[int, str]) -> None:
    """Rasterise room outlines into ``map_data.room_pixels`` (id << 2 | 1, same grid as the map).

    Sub-type 1 makes ``render_map_png`` tint only floor pixels, so walls inside an
    outline still draw as walls.
    """
    mask = Image.new("L", (map_data.width, map_data.height), 0)
    draw = ImageDraw.Draw(mask)
    for chain in chains:
        if len(chain.cells) >= 3:
            draw.polygon(chain.cells, fill=chain.room_id << 2 | 1, outline=chain.room_id << 2 | 1)
    map_data.room_pixels = mask.tobytes()
    map_data.room_outline_width = map_data.width
    map_data.room_outline_height = map_data.height
    map_data.room_outline_origin_x = map_data.origin_x
    map_data.room_outline_origin_y = map_data.origin_y
    map_data.room_names = names


def decode_pose(payload: bytes) -> tuple[int, int, float]:
    """Robot pose from a type-7 payload: (x [cm], y [cm], heading [rad])."""
    values = {1: 0, 2: 0, 3: 0}
    pose = _fields(_blobs(_fields(undelimit(payload)), 1)[0])
    for field_no in pose:
        zigzag = _varint(pose, field_no)
        values[field_no] = (zigzag >> 1) ^ -(zigzag & 1)
    return values[1], values[2], values[3] / 100


def _sign_magnitude(v: int) -> int:
    return -(v & 0x7FFF) if v & 0x8000 else v


def decode_path(payload: bytes) -> list[tuple[int, int]]:
    """Path points [cm] from a type-3 payload; the per-point flag byte is not used."""
    # iter_unpack raises on a trailing partial record, so trim to whole 5 B records.
    return [
        (_sign_magnitude(x), _sign_magnitude(y))
        for x, y, _flag in struct.iter_unpack(">HHB", payload[:len(payload) - len(payload) % 5])
    ]


def decode_history(blob: bytes) -> tuple[MapData, list[RoomChain], list[tuple[int, str]]]:
    """Decode a per-clean history ``.bin`` from Tuya storage into map, room outlines and names."""
    top = _fields(blob)
    return decode_map(_blobs(top, 10)[0]), decode_rooms(_blobs(top, 11)[0]), decode_room_names(_blobs(top, 3)[0])

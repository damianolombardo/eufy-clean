"""Unit tests for api/tuya_map_decode.py: m/m/i frame parsing and L60 map/room/pose/path decoding."""
import struct
import zlib

import pytest

from custom_components.robovac_mqtt.api.tuya_map_decode import (
    FRAME_MAP,
    RoomChain,
    apply_rooms,
    decode_history,
    decode_map,
    decode_path,
    decode_pose,
    decode_room_names,
    decode_rooms,
    name_rooms,
    parse_frame,
)
from custom_components.robovac_mqtt.utils import encode_varint


def _lz4_literals(data: bytes) -> bytes:
    """Encode *data* as a single literal-only LZ4 block (valid, just uncompressed)."""
    n = len(data)
    if n < 15:
        return bytes([n << 4]) + data
    rest = n - 15
    ext = b"\xff" * (rest // 255) + bytes([rest % 255])
    return b"\xf0" + ext + data


def _len_field(field_no: int, value: bytes) -> bytes:
    return encode_varint(field_no << 3 | 2) + encode_varint(len(value)) + value


def _int_field(field_no: int, value: int) -> bytes:
    return encode_varint(field_no << 3) + encode_varint(value)


def _packed(field_no: int, blob: bytes) -> bytes:
    """LZ4(zlib(blob)) in *field_no* with its decompressed size in the next field."""
    z = zlib.compress(blob)
    return _len_field(field_no, _lz4_literals(z)) + _int_field(field_no + 1, len(z))


def _map_message(width: int, height: int, cells: bytes) -> bytes:
    header = struct.pack("<15I5fI", *([0] * 13), width, height, -1.5, -2.0, 8.0, 9.0, 0.05, len(cells))
    return _packed(4, header + cells) + _len_field(7, b"Floor")


def _rooms_message(chains: list[tuple[int, list[tuple[int, int, int]]]]) -> bytes:
    blob = struct.pack("<II", 0, len(chains))
    for room_id, cells in chains:
        blob += struct.pack("<II", room_id, len(cells))
        blob += b"".join(struct.pack("<III", *c) for c in cells)
    return _int_field(1, 2) + _packed(7, blob)


def _names_message(names: list[tuple[int, str]]) -> bytes:
    return b"".join(_len_field(2, _int_field(1, i) + _len_field(2, n.encode())) for i, n in names)


def _frame(kind: int, payload: bytes, seq: int = 0) -> bytes:
    head = b"\x55\xaa\x00\x00\x00\x02" + struct.pack(">H", len(payload) + 17) + b"\x00\x01"
    head += bytes([kind, 0, 0]) + struct.pack(">IIH", 1_700_000_000, seq, len(payload))
    return head + payload + b"\x00\x00\x00\x00"


def _delimited(message: bytes) -> bytes:
    return encode_varint(len(message)) + message


def _zigzag(v: int) -> int:
    return (v << 1) ^ (v >> 31)


SQUARE = [(1, 1, 0), (4, 1, 0), (4, 4, 0), (1, 4, 0)]


def test_parse_frame_splits_header_and_payload():
    frame = parse_frame(_frame(FRAME_MAP, b"abc", seq=1320))
    assert (frame.kind, frame.sub, frame.ts, frame.seq, frame.payload) == (2, 0, 1_700_000_000, 1320, b"abc")


@pytest.mark.parametrize("raw", [b"", b"{\"protocol\":4}" + b"\x00" * 20])
def test_parse_frame_rejects_non_frames(raw):
    assert parse_frame(raw) is None


def test_decode_map_repacks_grid_and_converts_origin_to_cm():
    cells = bytes([0, 127, 128, 127, 0, 0])
    map_data = decode_map(_map_message(3, 2, cells))
    assert (map_data.width, map_data.height) == (3, 2)
    assert (map_data.origin_x, map_data.origin_y, map_data.resolution) == (-150, -200, 5)
    pixels = [(map_data.raw_pixels[i >> 2] >> ((i & 3) * 2)) & 3 for i in range(6)]
    assert pixels == [0, 2, 1, 2, 0, 0]


def test_decode_rooms_returns_outline_cells_without_door_ids():
    chains = decode_rooms(_rooms_message([(10, [(1, 1, 0), (2, 1, 3), (2, 2, 0)]), (11, SQUARE)]))
    assert [c.room_id for c in chains] == [10, 11]
    assert chains[0].cells == [(1, 1), (2, 1), (2, 2)]


def test_decode_room_names_keeps_source_order():
    assert decode_room_names(_names_message([(6, "Landing"), (2, "Study")])) == [(6, "Landing"), (2, "Study")]


@pytest.mark.parametrize(
    ("names", "expected"),
    [([(6, "Hall"), (2, "Study")], {15: "Hall", 11: "Study"}), ([(6, "Hall")], {})],
)
def test_name_rooms_pairs_by_position_only_when_lengths_match(names, expected):
    chains = [RoomChain(15, []), RoomChain(11, [])]
    assert name_rooms(chains, names) == expected


def test_apply_rooms_fills_outline_with_room_id_and_floor_subtype():
    map_data = decode_map(_map_message(6, 6, bytes([127]) * 36))
    apply_rooms(map_data, [RoomChain(12, [(x, y) for x, y, _ in SQUARE])], {12: "Study"})
    assert map_data.room_pixels[2 * 6 + 2] == 12 << 2 | 1
    assert map_data.room_pixels[0] == 0
    assert map_data.room_names == {12: "Study"}
    assert map_data.room_outline_width == 6


def test_decode_history_pairs_embedded_map_rooms_and_names():
    blob = (
        _len_field(3, _names_message([(1, "Study")]))
        + _len_field(10, _map_message(6, 6, bytes(36)))
        + _len_field(11, _rooms_message([(10, SQUARE)]))
    )
    map_data, chains, names = decode_history(blob)
    assert map_data.width == 6
    assert name_rooms(chains, names) == {10: "Study"}


def test_decode_pose_unzigzags_cm_and_centiradians():
    inner = _int_field(1, _zigzag(-120)) + _int_field(3, _zigzag(157))
    assert decode_pose(_delimited(_len_field(1, inner))) == (-120, 0, 1.57)


def test_decode_path_reads_sign_magnitude_records_and_ignores_partial_tail():
    payload = struct.pack(">HHB", 0x8000 | 25, 300, 0) + struct.pack(">HHB", 7, 0x8000 | 8, 12) + b"\x01\x02"
    assert decode_path(payload) == [(-25, 300), (7, -8)]

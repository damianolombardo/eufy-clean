"""Tuya mobile-app MQTT session carrying the ``m/m/i/<devId>`` live-map stream.

Tuya-cloud models (L60 SES) stream map, rooms, pose and path to the app over the
app's own Tuya MQTT broker rather than eufy biz/. Subscribing is enough to receive
frames; no request is sent. One session serves every device on the account.

Credentials come from the Tuya mobile login (``tuya.m.user.uid.password.login``):
username = ``sid``, password = md5hex(``ecode``)[8:24], and the client id mirrors
the eufy app's Tuya SDK format.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import ssl
import uuid
from collections.abc import Callable
from functools import partial

from paho.mqtt import client as mqtt

_LOGGER = logging.getLogger(__name__)

_APP_PACKAGE = "com.oceanwing.battery.cam"
_CLIENT_ID_SALT = "sdkfasodifca"
_TOPIC_PREFIX = "m/m/i/"


def _md5(data: str) -> str:
    return hashlib.md5(data.encode()).hexdigest()


def _tls_context() -> ssl.SSLContext:
    return ssl.create_default_context()


class TuyaMapStream:
    """Account-wide Tuya MQTT session that fans ``m/m/i`` frames out per device."""

    def __init__(self, host: str, port: int, uid: str, sid: str, ecode: str) -> None:
        self.host = host
        self.port = port
        self._client_id = f"{_APP_PACKAGE}_mb_{uuid.uuid4().hex[:16]}_{_md5(uid + _CLIENT_ID_SALT)}"
        self._username = sid
        self._password = _md5(ecode)[8:24]
        self._callbacks: dict[str, Callable[[bytes], None]] = {}
        self._mqtt_client: mqtt.Client | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self, dev_id: str, callback: Callable[[bytes], None]) -> None:
        """Route frames for *dev_id* to *callback* (called on the event loop)."""
        self._callbacks[dev_id] = callback
        if self._mqtt_client and self._mqtt_client.is_connected():
            self._mqtt_client.subscribe(_TOPIC_PREFIX + dev_id)

    def resubscribe(self, dev_id: str) -> None:
        """Renew the device subscription; the device only starts streaming for a
        subscription made while it is active, so call this when a clean starts."""
        if self._mqtt_client and self._mqtt_client.is_connected():
            self._mqtt_client.unsubscribe(_TOPIC_PREFIX + dev_id)
            self._mqtt_client.subscribe(_TOPIC_PREFIX + dev_id)

    async def connect(self) -> None:
        """Open the TLS session; subscriptions are (re)made on every connect."""
        self._loop = asyncio.get_running_loop()
        client = mqtt.Client(client_id=self._client_id, transport="tcp")
        client.username_pw_set(self._username, self._password)
        client.tls_set_context(await self._loop.run_in_executor(None, _tls_context))
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        self._mqtt_client = client
        _LOGGER.debug("Connecting Tuya map stream to %s:%d", self.host, self.port)
        await self._loop.run_in_executor(None, partial(client.connect, self.host, self.port, 60))
        client.loop_start()

    async def disconnect(self) -> None:
        """Stop the network loop and close the session."""
        if not self._mqtt_client:
            return
        client, self._mqtt_client = self._mqtt_client, None
        client.loop_stop()
        loop = self._loop or asyncio.get_running_loop()
        await loop.run_in_executor(None, client.disconnect)

    def _on_connect(self, client: mqtt.Client, userdata: object, flags: dict, rc: int) -> None:
        if rc != 0:
            _LOGGER.error("Tuya map stream connect refused, rc=%d", rc)
            return
        _LOGGER.debug("Tuya map stream connected; subscribing %d device(s)", len(self._callbacks))
        for dev_id in self._callbacks:
            client.subscribe(_TOPIC_PREFIX + dev_id)

    @staticmethod
    def _on_disconnect(client: mqtt.Client, userdata: object, rc: int) -> None:
        if rc != 0:
            _LOGGER.warning("Tuya map stream disconnected unexpectedly, rc=%d", rc)

    def _on_message(self, client: mqtt.Client, userdata: object, msg: mqtt.MQTTMessage) -> None:
        callback = self._callbacks.get(msg.topic.removeprefix(_TOPIC_PREFIX))
        if callback and self._loop:
            self._loop.call_soon_threadsafe(callback, msg.payload)

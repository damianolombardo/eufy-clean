"""Unit tests for api/tuya_map_stream.py: credentials and m/m/i frame routing."""
import hashlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from custom_components.robovac_mqtt.api.tuya_map_stream import TuyaMapStream


def _stream():
    return TuyaMapStream("m1.example.com", 8883, "eu123", "sid-1", "ecode-1")


def test_credentials_follow_tuya_app_recipe():
    stream = _stream()
    assert stream._username == "sid-1"
    assert stream._password == hashlib.md5(b"ecode-1").hexdigest()[8:24]
    assert stream._client_id.startswith("com.oceanwing.battery.cam_mb_")
    assert stream._client_id.endswith(hashlib.md5(b"eu123sdkfasodifca").hexdigest())


def test_connect_subscribes_every_registered_device():
    stream = _stream()
    stream.subscribe("dev_a", MagicMock())
    stream.subscribe("dev_b", MagicMock())
    client = MagicMock()
    stream._on_connect(client, None, {}, 0)
    assert [c.args[0] for c in client.subscribe.call_args_list] == ["m/m/i/dev_a", "m/m/i/dev_b"]


def test_message_routes_to_device_callback_on_loop():
    stream = _stream()
    stream._loop = MagicMock()
    callback = MagicMock()
    stream.subscribe("dev_a", callback)
    stream._on_message(None, None, SimpleNamespace(topic="m/m/i/dev_a", payload=b"\x55\xaa"))
    stream._on_message(None, None, SimpleNamespace(topic="m/m/i/other", payload=b"x"))
    stream._loop.call_soon_threadsafe.assert_called_once_with(callback, b"\x55\xaa")


def test_resubscribe_renews_device_subscription():
    stream = _stream()
    stream._mqtt_client = MagicMock()
    stream.resubscribe("dev_a")
    stream._mqtt_client.unsubscribe.assert_called_once_with("m/m/i/dev_a")
    stream._mqtt_client.subscribe.assert_called_once_with("m/m/i/dev_a")

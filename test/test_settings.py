
from config import settings

def test_settings():
    assert settings.url is not None
    assert settings.yahoo.url is not None
    assert settings.email is not None
    assert settings.user.password is not None
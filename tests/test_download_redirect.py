import requests

TEST_URL = "https://dts.podtrac.com/redirect.mp3/pdst.fm/e/pfx.vpixl.com/6qj4J/pscrb.fm/rss/p/nyt.simplecastaudio.com/03d8b493-87fc-4bd1-931f-8a8e9b945d8a/episodes/7b5d7003-dcf0-4d08-bf6b-1b93f4fedb85/audio/128/default.mp3"


def test_redirect():
    """Test that the URL can be downloaded with increased redirect limit"""
    session = requests.Session()
    session.headers.clear()
    session.max_redirects = 60

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "audio/*,*/*;q=0.9",
        "Accept-Language": "en-US,en;q=0.9",
    }

    print(f"Testing download from: {TEST_URL}")
    print(f"Max redirects: {session.max_redirects}")

    try:
        response = session.get(
            TEST_URL, stream=True, timeout=300, headers=headers, allow_redirects=True
        )
        response.raise_for_status()

        content_length = response.headers.get("content-length", "unknown")
        print(f"✅ Request successful!")
        print(f"   Status code: {response.status_code}")
        print(f"   Content-Length: {content_length}")

        # Download a small chunk to verify it's working
        chunk = next(response.iter_content(chunk_size=8192), None)
        if chunk:
            print(f"   First chunk: {len(chunk)} bytes")
            print(f"   First bytes (hex): {chunk[:16].hex()}")

        return True
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    success = test_redirect()
    sys.exit(0 if success else 1)

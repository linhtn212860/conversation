#!/usr/bin/env python3
"""
Quick test to verify client-server connection
"""
import asyncio
import websockets
import json

async def test():
    print("Testing connection to ws://localhost:8765/ws")
    print("Make sure server is running first!")
    print()

    try:
        print("Connecting...")
        ws = await asyncio.wait_for(
            websockets.connect('ws://localhost:8765/ws'),
            timeout=10.0
        )
        print("✓ Connected!")

        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(msg)
        print(f"✓ Received: {data}")

        # Try sending a ping
        await ws.send(json.dumps({"type": "ping"}))
        print("✓ Sent ping")

        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
        data = json.loads(msg)
        print(f"✓ Received: {data}")

        await ws.close()
        print()
        print("✓ Connection test PASSED!")

    except asyncio.TimeoutError:
        print("✗ Timeout! Server not responding.")
    except ConnectionRefusedError:
        print("✗ Connection refused. Is server running?")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())

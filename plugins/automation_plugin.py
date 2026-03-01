def register():
    async def turn_on_light(args: dict):
        room = args.get("room", "unknown")
        msg = f"[automation_plugin] ACTION: turn_on_light -> room={room}"
        print(msg)
        return {"status": "ok", "message": msg}

    async def turn_off_light(args: dict):
        room = args.get("room", "unknown")
        msg = f"[automation_plugin] ACTION: turn_off_light -> room={room}"
        print(msg)
        return {"status": "ok", "message": msg}

    return {
        "turn_on_light": turn_on_light,
        "turn_off_light": turn_off_light,
    }

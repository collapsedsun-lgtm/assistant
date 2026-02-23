def register():
    async def turn_on_light(args: dict):
        room = args.get("room", "unknown")
        return {"status": "ok", "message": f"(example plugin) turned on {room}"}

    async def turn_off_light(args: dict):
        room = args.get("room", "unknown")
        return {"status": "ok", "message": f"(example plugin) turned off {room}"}

    async def get_time(args: dict):
        return {"status": "ok", "message": "This plugin does not implement real time."}

    return {
        "turn_on_light": turn_on_light,
        "turn_off_light": turn_off_light,
        "get_time": get_time,
    }

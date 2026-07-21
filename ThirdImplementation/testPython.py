import subprocess
import json
import os
import asyncio
import shutil
import websockets

current_dir = os.path.dirname(os.path.abspath(__file__))
prolog_file = os.path.join(current_dir, "test.pl")

print(prolog_file)

def run_scasp(query: str):
    result = subprocess.run(
        ["swipl", prolog_file, query],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"swipl failed: {result.stderr}")
    return json.loads(result.stdout.strip())

async def handler(socket):
    shutil.copy("rules.pl", "rules_temp.pl")
    with open("rules_temp.pl", "a") as factsFile:
        print("client connected!")
        try:
            async for message in socket:
                jsonMessage = json.loads(message)

                if(jsonMessage["message_type"] == "heard_noise"):
                    factsFile.write("noise(unknown).\n")
                elif(jsonMessage["message_type"] == "vase_broken"):
                    factsFile.write(f"broken_vase({jsonMessage['culprit']}).\n")
                elif(jsonMessage["message_type"] == "suspicious_sighting"):
                    factsFile.write(f"suspicious_sighting(player).\n")
                    factsFile.write("player_in_restricted_area.\n")
                elif(jsonMessage["message_type"] == "diamond_broken"):
                    factsFile.write("diamond_saw_broken.\n")
                elif(jsonMessage["message_type"] == "alarm_raised"):
                    factsFile.write("alarm_raised.\n")
                elif(jsonMessage["message_type"] == "get_action"):
                    returnedDict = run_scasp("chosen_action(X)")
                    returnedArr = returnedDict["solutions"]
                    solutionArr = []
                    for solution in returnedArr:
                        appendedValue = solution["bindings"]["X"]
                        solutionArr.append(appendedValue)
                    await socket.send(json.dumps(solutionArr))
                else:
                    raise ValueError(f"json message is invalid, got {jsonMessage['message_type']}")
                
                factsFile.flush()


        except websockets.ConnectionClosed:
            print("Client disconnected")

async def mainTask():
    async with websockets.serve(handler, "localhost", 6767):
        print("WebSocket server running!")
        await asyncio.Future()

asyncio.run(mainTask())
from word_suggestion import *
import asyncio, websockets, json

#TODO take in CLI argument for the user

def main():
    async def get_text(websocket, path):
        text = await websocket.recv()
        print(f"< {text}")
        suggestions = word_suggest.predict(text)
        await websocket.send(json.dumps(suggestions))
    
    model_metadata = Suggest_Util.load_dict()
    word_suggest = Word_Suggestion(*model_metadata.values())
    word_suggest.load_and_build_latest_model()
    print(word_suggest.predict("they"))
    HOST = "localhost"
    PORT = 50007

    start_server = websockets.serve(get_text, HOST, PORT)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()


    





if __name__ == "__main__":
    main()
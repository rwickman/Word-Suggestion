from word_suggestion import Word_Suggestion, Suggest_Util
import asyncio, websockets, json
from lsh import LSH

#TODO take in CLI argument for the user

def main():
    async def get_response(websocket, path):
        # try:
        response = await websocket.recv()
        response_decoded = json.loads(response)
        text = response_decoded["text"]
        print(f"< {text}")
        response_class = response_decoded["class"]

        if response_class:
            lsh_data = lsh.get_data(response_class)
            word_suggest.train_update(lsh_data)

        suggestions = word_suggest.predict(text)
        lsh_suggestions = lsh.query_and_predict(text)
        await websocket.send(json.dumps(
        {"suggestions" : suggestions,
            "lsh suggestions" : lsh_suggestions }
            ))
        # except Exception as e:
        #     print("ERROR IN ASYNC RESPONSE: ", e)


    
    model_metadata = Suggest_Util.load_dict()
    word_suggest = Word_Suggestion(*model_metadata.values())
    word_suggest.load_and_build_latest_model()
    lsh = LSH()
    print(word_suggest.predict("they"))
    HOST = "localhost"
    PORT = 50007

    start_server = websockets.serve(get_response, HOST, PORT)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()

if __name__ == "__main__":
    main()
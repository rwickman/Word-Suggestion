from word_suggestion import Word_Suggestion, Suggest_Util
import asyncio, websockets, json
from lsh import LSH
import tensorflow as tf
import argparse

def main(args):
    async def get_response(websocket, path):
        # try:
        response = await websocket.recv()
        response_decoded = json.loads(response)
        text = response_decoded["text"]
        print(f"< {text}")
        response_class = response_decoded["class"]

        if args.find_similar and args.update and response_class:
            lsh_data = lsh.get_data(response_class)
            # t = threading.Thread(target=word_suggest.train_update, args=(lsh_data, ))
            word_suggest.train_update(lsh_data)
            # t.start()

        suggestions = word_suggest.predict(text)
        lsh_suggestions = []
        if args.find_similar:
            lsh_suggestions = lsh.query_and_predict(text)
        await websocket.send(json.dumps(
        {"suggestions" : suggestions,
            "lsh suggestions" : lsh_suggestions }
            ))  
        # except Exception as e:
        #     print("ERROR IN ASYNC RESPONSE: ", e)

    if args.update:
        model_metadata = Suggest_Util.load_dict("../config/updated_model_metadata.json")
        word_suggest = Word_Suggestion(*model_metadata.values())
        word_suggest.build_model()
        word_suggest.model =  tf.keras.models.load_model('../models/training_checkpoints/conversation/conv_model.h5') #load_and_build_latest_model()
    else:
        model_metadata = Suggest_Util.load_dict("../config/model_metadata.json")
        word_suggest = Word_Suggestion(*model_metadata.values())
        word_suggest.build_model()
        word_suggest.model =  tf.keras.models.load_model('../models/training_checkpoints/conversation_no_update/conv_model.h5')

    if args.find_similar:
        lsh = LSH()
    
    print(word_suggest.predict("they"))
    HOST = "localhost"
    PORT = 50007

    start_server = websockets.serve(get_response, HOST, PORT)
    loop = asyncio.get_event_loop()
    print("READY TO SERVE PREDICTIONS. YOU MAY NOW START UP WITH WEBPAGE.")
    loop.run_until_complete(start_server)
    loop.run_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Word Suggestion Demo")
    parser.add_argument('--update', action="store_true", help="Update the users RNN")
    parser.add_argument('--find-similar', action="store_true", help="Find similar users")
    args = parser.parse_args()
    main(args)
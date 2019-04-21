from word_suggestion import Word_Suggestion, Suggest_Util
import sys

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "movie":
        quotes, max_sequence_length = Suggest_Util.parse_movie_quotes("../data/movie_quotes.json")
        word_to_id, id_to_word = Suggest_Util.words_to_id(quotes, True)
        data = Suggest_Util.split_by_word_sequentially(quotes, word_to_id, max_sequence_length + 1)
        vocab_size = len(word_to_id)
        word_suggest = Word_Suggestion(vocab_size, max_sequence_length, word_to_id, id_to_word, '../models/training_checkpoints/movie')
        word_suggest.build_model()
        word_suggest.train(data, "", num_epochs=25, has_checkpoint=True)
        print(word_suggest.predict("mama always said"))

        model_metadata = {"vocab_size" : vocab_size, "max_sequence_length" : max_sequence_length, "word_to_id" : word_to_id, "id_to_word" : id_to_word, "checkpoint_dir" :'../models/training_checkpoints/movie'}
        Suggest_Util.save_dict(model_metadata, "../config/movie_model_metadata.json")
    elif len(sys.argv) > 1 and sys.argv[1] == "lyrics":
        lyrics, max_sequence_length = Suggest_Util.parse_movie_quotes("../data/lyrics.json", "Lyrics")
        word_to_id, id_to_word = Suggest_Util.words_to_id(lyrics, True)
        vocab_size = len(word_to_id)
        word_suggest = Word_Suggestion(vocab_size, max_sequence_length, word_to_id, id_to_word, '../models/training_checkpoints/lyrics')
        word_suggest.build_model()
        word_suggest.train(lyrics,"lyrics", num_epochs=25, has_checkpoint=True)
        print(word_suggest.predict("drink whiskey and red"))

        model_metadata = {"vocab_size" : vocab_size, "max_sequence_length" : max_sequence_length, "word_to_id" : word_to_id, "id_to_word" : id_to_word, "checkpoint_dir" : "../models/training_checkpoints/lyrics"}
        Suggest_Util.save_dict(model_metadata, "../config/lyrics_model_metadata.json") 
    else:
        data, max_sequence_length = Suggest_Util.parse_conversation_json("../data/conversation.json", 100)
        data = Suggest_Util.clean_data(data)
        word_to_id, id_to_word = Suggest_Util.words_to_id(data)
        vocab_size = len(word_to_id)
        word_suggest = Word_Suggestion(vocab_size, max_sequence_length, word_to_id, id_to_word)
        word_suggest.build_model()
    
        word_suggest.train(data,"conversation", 45, True)

        model_metadata = {"vocab_size" : vocab_size, "max_sequence_length" : max_sequence_length, "word_to_id" : word_to_id, "id_to_word" : id_to_word, "checkpoint_dir" : '../models/training_checkpoints/conversation'}
        Suggest_Util.save_dict(model_metadata)  
    
    # print(word_suggest.generate_seq("they", 20))
    # print(word_suggest.predict(""))
    # print(word_suggest.predict("they do"))
    # print(word_suggest.predict("let's")) 
    # print(word_suggest.predict("i hope")) #so
    # print(word_suggest.predict("you're gonna"))
    
if __name__ == "__main__":
    main()
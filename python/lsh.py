from word_suggestion import Word_Suggestion, Suggest_Util 
from datasketch import MinHashLSHEnsemble, MinHash
import pickle

class lsh:
    def __init__(self):
        
        self.movies = Suggest_Util.load_dict("../data/movie_quotes.json")
        self.lyrics = Suggest_Util.load_dict("../data/lyrics.json")
        
        self.movies_metadata = Suggest_Util.load_dict("../config/movie_model_metadata.json")
        self.lyrics_metadata = Suggest_Util.load_dict("../config/lyrics_model_metadata.json")
        self.lsh_metdata = Suggest_Util.load_dict("../config/lsh_metadata.json")

        self.word_suggest_movies = Word_Suggestion(*self.movies_metadata.values())
        self.word_suggest_lyrics = Word_Suggestion(*self.lyrics_metadata.values())

        self.word_suggest_movies.load_and_build_latest_model()
        self.word_suggest_lyrics.load_and_build_latest_model()

        self.lsh_ensemble = pickle.load(open("../models/lshensemble.p", "rb"))

        # This will store the keys of the results with containment > 0.64: 
        self.query_results = {"movies" : set(), "lyrics" : set()}

    def create_lsh_ensemble(self):
        minhashes = []
        lsh_metadata = {"movies" : [], "lyrics" : []}
        for movie in self.movies:
            minhash_key = movie["movie"] + " quote: " + movie["quote"]
            clean_quote = Suggest_Util.clean_data(movie["quote"])
            minhash_set = set(clean_quote.split())
            minhash = MinHash(num_perm=128)
            for word in minhash_set:
                minhash.update(word.encode("utf8"))
            minhashes.append( (minhash_key, minhash, len(minhash_set)) )
            lsh_metadata["movies"].append(minhash_key)

        for lyric in self.lyrics:
            minhash_key = lyric["Artist"] + " song: " + lyric["Song"]
            clean_lyric = Suggest_Util.clean_data(lyric["Lyrics"])
            minhash_set = set(clean_lyric.split())
            minhash = MinHash(num_perm=128)
            for word in minhash_set:
                minhash.update(word.encode("utf8"))
            minhashes.append( (minhash_key, minhash, len(minhash_set)) )
            lsh_metadata["lyrics"].append(minhash_key)

        # Create an LSH Ensemble index with threshold and number of partition
        # settings.
        lshensemble = MinHashLSHEnsemble(threshold=0.64, num_perm=128, num_part=32)

        # Index takes an iterable of (key,
        #  minhash, size)
        lshensemble.index(minhashes)

        pickle.dump(lshensemble, open("../models/lshensemble.p", "wb"))
        Suggest_Util.save_dict(lsh_metadata, "../config/lsh_metadata.json")

    def query_and_predict(self, query):
        self.query_results = {"movies" : set(), "lyrics" : set()}
        query_minhash = MinHash(num_perm=128)
        clean_query = Suggest_Util.clean_data(query)
        query_set = set(clean_query.split())
        for word in query_set:
            query_minhash.update(word.encode("utf8"))
        for key in self.lsh_ensemble.query(query_minhash, len(query_set)):
            if key in self.lsh_metdata["movies"]:
                self.query_results["movies"].add(key)
            elif key in self.lsh_metdata["lyrics"]:
                self.query_results["lyrics"].add(key)
        
        if len(self.query_results["movies"]) > 0:
            movie_predictions = self.word_suggest_movies.predict(query)
        if len(self.query_results["lyrics"]) > 0:
            lyric_predictions = self.word_suggest_lyrics.predict(query)
        
        return {"movie predictions" : movie_predictions, "lyric predictions" : lyric_predictions}

lsh = lsh()
print(lsh.query_and_predict("can still shut down a party I can hang with anybody"))
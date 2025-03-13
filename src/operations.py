from Preprocessing import Preprocessing
import operator

columns_to_remove_model = ['title', 'backdrop_path', 'homepage',
    'imdb_id', 'original_title',
    'poster_path', 'tagline', 'budget',
    'revenue',
    'keywords']

operations_model = [
    {"type": Preprocessing.clean, "params": {'col': columns_to_remove_model, 'drop': True}},
    {"type": Preprocessing.filter, "params": {'col': 'status', 'value': 'Released', 'operator': operator.eq}},
    {"type": Preprocessing.filter, "params": {'col': 'vote_count', 'value': 20, 'operator': operator.ge}},
    {"type": Preprocessing.date_manipulation, "params": {'col': 'release_date'}},
    {"type": Preprocessing.one_hot_encoding_list, "params": {'col': 'genres', 'string_to_list': True}},
    {"type": Preprocessing.frequency_encoding, "params": {'col': 'original_language'}},
    {"type": Preprocessing.frequency_encoding_list, "params": {'col': 'production_companies', 'string_to_list': True}},
    {"type": Preprocessing.frequency_encoding_list, "params": {'col': 'production_countries', 'string_to_list': True}},
    {"type": Preprocessing.frequency_encoding_list, "params": {'col': 'spoken_languages', 'string_to_list': True}},
    {"type": Preprocessing.clean, "params": {'col': ['status', 'id', 'overview', 'popularity', 'vote_count'], 'drop': True}}
]

columns_to_keep_sbert = ['title', 'release_date', 'overview']

operations_sbert = [

    {"type": Preprocessing.filter, "params": {'col': 'status', 'value': 'Released', 'operator': operator.eq}},
    {"type": Preprocessing.clean, "params": {'col': columns_to_keep_sbert, 'drop': False}}
]
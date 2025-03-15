from Preprocessing import Preprocessing
import operator

columns_to_remove_model = ['title', 'backdrop_path', 'homepage', 'original_title',
    'poster_path', 'tagline', 'budget',
    'revenue',
    'keywords']

operations_model = [
    {"type": Preprocessing.clean, "params": {'col': columns_to_remove_model, 'drop': True, 'duplicates_to_check': ['imdb_id']}},
    {"type": Preprocessing.filter, "params": {'col': 'status', 'value': 'Released', 'operator': operator.eq}},
    {"type": Preprocessing.filter, "params": {'col': 'vote_count', 'value': 20, 'operator': operator.ge}},
    {"type": Preprocessing.date_manipulation, "params": {'col': 'release_date'}},
    {"type": Preprocessing.one_hot_encoding_list, "params": {'col': 'genres', 'string_to_list': True}},
    {"type": Preprocessing.frequency_encoding, "params": {'col': 'original_language'}},
    {"type": Preprocessing.frequency_encoding_list, "params": {'col': 'production_companies', 'string_to_list': True}},
    {"type": Preprocessing.frequency_encoding_list, "params": {'col': 'production_countries', 'string_to_list': True}},
    {"type": Preprocessing.frequency_encoding_list, "params": {'col': 'spoken_languages', 'string_to_list': True}},
    {"type": Preprocessing.clean, "params": {'col': ['status', 'id', 'overview', 'popularity', 'vote_count', 'imdb_id'], 'drop': True, 'duplicates_to_check': None}}
]

columns_to_keep_sbert = ['title', 'release_date', 'overview', 'imdb_id']

operations_sbert = [

    {"type": Preprocessing.filter, "params": {'col': 'status', 'value': 'Released', 'operator': operator.eq}},
    {"type": Preprocessing.clean, "params": {'col': columns_to_keep_sbert, 'drop': False, 'duplicates_to_check': ['imdb_id']}}
]
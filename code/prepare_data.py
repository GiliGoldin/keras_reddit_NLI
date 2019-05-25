import os
import random
import sys
import pickle
from keras.preprocessing.text import one_hot
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from RedditUtils import Utils
from RedditConfig import RedditConfig
from my_reddit_classifier import config, Classifier, PERCENT_FOR_TEST_SET, get_features_names_by_features_combinations, set_users_dicts_path
CALCULATE_VOCABULARY_SIZE = False
VOCABULARY_SIZE = 5000#actual size is about 100,000 but using only 5000 improves results
NUMBER_OF_LANGUAGES = 30
NUMBER_OF_FAMILIES = 5
NUMBER_OF_BINARY = 2
CREATE_NEW_DATA = False
SPELLING = False
features_combination = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0)#relevant only for spelling features
from RedditConfig import EUROPE_AND_NON_EUROPE_USERS_DICTS_PATH, ONLY_EUROPE_USERS_DICTS_PATH, ALL_LANGS_FLAG, FAMILIES_FLAG, BINARY_FLAG

class Data_preparation:
    def get_vocabulary_size(self, train_set, test_set):
        vocabulary = set()
        voc_size = VOCABULARY_SIZE
        if CALCULATE_VOCABULARY_SIZE:
            for chunk in train_set:
                chunks_words = chunk.split()
                vocabulary.update(chunks_words)
            for chunk in test_set:
                chunks_words = chunk.split()
                vocabulary.update(chunks_words)
                voc_size = len(vocabulary)
                print(voc_size)
                print("double vocabulary size" + str(voc_size))
        return voc_size


    def prepare_data(self, config):
        classifier = Classifier()
        tokenizer_file_name = ""
        if config.only_europe:
            all_users_dicts_path = os.path.join('..',ONLY_EUROPE_USERS_DICTS_PATH)
            eurupe_users_dict_name = 'USERS_DICT_EUROPE.pkl'
            users_per_label_dict_europe_name = 'users_per_label_dict_europe.pkl'
        else:
            all_users_dicts_path = os.path.join('..',EUROPE_AND_NON_EUROPE_USERS_DICTS_PATH)
            eurupe_users_dict_name = 'USERS_DICT_FULL_EUROPE.pkl'
            users_per_label_dict_europe_name = 'users_per_label_dict_full_europe.pkl'

        if config.do_all_languages:
            users_dicts_path = set_users_dicts_path(all_users_dicts_path, ALL_LANGS_FLAG)
            number_of_classes = NUMBER_OF_LANGUAGES
        elif config.do_families:
            users_dicts_path = set_users_dicts_path(all_users_dicts_path, FAMILIES_FLAG)
            number_of_classes = NUMBER_OF_FAMILIES
        else:
            users_dicts_path = set_users_dicts_path(all_users_dicts_path, BINARY_FLAG)
            number_of_classes = NUMBER_OF_BINARY

        users_dict_europe_path = os.path.join(users_dicts_path, eurupe_users_dict_name)
        users_dict_non_europe_path = os.path.join(users_dicts_path, 'NON_EUROPE_USERS_DICT.pkl')
        users_per_label_dict_path = os.path.join(users_dicts_path, users_per_label_dict_europe_name)

        if CREATE_NEW_DATA:
            with open(users_dict_europe_path, 'rb') as f:
                USERS_DICT = pickle.load(f)

            with open(users_per_label_dict_path, 'rb') as f:
                users_per_label_dict_europe = pickle.load(f)


        if config.only_europe:
            tokenizer_file_name = 'europe_tokenizer.pkl'
            if CREATE_NEW_DATA:
                train_set, test_set, train_labels, test_labels, test_users, train_users = classifier.split_data_to_train_set_and_test_set(
                        PERCENT_FOR_TEST_SET, users_per_label_dict_europe, USERS_DICT)
                # VOCABULARY_SIZE = self.get_vocabulary_size(train_set, test_set)
                tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)#, oov_token='(~?~)')
                tokenizer.fit_on_texts(train_set)
                with open(tokenizer_file_name, 'wb') as f:
                    pickle.dump(tokenizer, f)
                with open('europe_train_set.pkl', 'wb') as f:
                    pickle.dump(train_set, f)
                with open('europe_train_labels.pkl', 'wb') as f:
                    pickle.dump(train_labels, f)
                with open('europe_test_set.pkl', 'wb') as f:
                    pickle.dump(test_set, f)
                with open('europe_test_labels.pkl', 'wb') as f:
                    pickle.dump(test_labels, f)

                if SPELLING:
                    test_spell_feature_values, test_spell_names = classifier.extract_features(test_set, test_users,
                                                                                  USERS_DICT,
                                                                                  features_combination)
                    train_spell_feature_values, train_spell_names  = classifier.extract_features(train_set, train_users, USERS_DICT,
                                                                                    features_combination)
                    with open('europe_test_spell_feature_values.pkl', 'wb') as f:
                        pickle.dump(test_spell_feature_values, f)
                    with open('europe_test_spell_names.pkl', 'wb') as f:
                        pickle.dump(test_spell_names, f)
                    with open('europe_train_spell_feature_values.pkl', 'wb') as f:
                        pickle.dump(train_spell_feature_values, f)
                    with open('europe_train_spell_names.pkl', 'wb') as f:
                        pickle.dump(train_spell_names, f)

            else:#load data from files
                with open('europe_train_set.pkl', 'rb') as f:
                    train_set = pickle.load(f)
                with open('europe_train_labels.pkl', 'rb') as f:
                    train_labels = pickle.load(f)
                with open('europe_test_set.pkl', 'rb') as f:
                    test_set = pickle.load(f)
                with open('europe_test_labels.pkl', 'rb') as f:
                    test_labels = pickle.load(f)
                if SPELLING:
                    with open('europe_test_spell_feature_values.pkl', 'rb') as f:
                        test_spell_feature_values = pickle.load(f)
                    with open('europe_test_spell_names.pkl', 'rb') as f:
                        test_spell_names = pickle.load(f)
                    with open('europe_train_spell_feature_values.pkl', 'rb') as f:
                        train_spell_feature_values = pickle.load(f)
                    with open('europe_train_spell_names.pkl', 'rb') as f:
                        train_spell_names = pickle.load(f)

        else:#Non europe
            tokenizer_file_name = 'non_europe_tokenizer.pkl'
            if CREATE_NEW_DATA:
                with open(users_dict_non_europe_path, 'rb') as f:
                    NON_EUROPE_USERS_DICT = pickle.load(f)
                train_set, test_set, train_labels, test_labels, test_users, train_users = classifier.split_to_europe_train_non_europe_test_with_different_users(
                    PERCENT_FOR_TEST_SET, users_per_label_dict_europe, USERS_DICT, NON_EUROPE_USERS_DICT)
                tokenizer = Tokenizer(num_words=VOCABULARY_SIZE, oov_token='outofvoc')
                tokenizer.fit_on_texts(train_set)
                with open(tokenizer_file_name, 'wb') as f:
                    pickle.dump(tokenizer, f)
                with open('non_europe_train_set.pkl', 'wb') as f:
                    pickle.dump(train_set, f)
                with open('non_europe_train_labels.pkl', 'wb') as f:
                    pickle.dump(train_labels, f)
                with open('non_europe_test_set.pkl', 'wb') as f:
                    pickle.dump(test_set, f)
                with open('non_europe_test_labels.pkl', 'wb') as f:
                    pickle.dump(test_labels, f)
                if SPELLING:
                    test_spell_feature_values, test_spell_names = classifier.extract_features(test_set, test_users, NON_EUROPE_USERS_DICT, features_combination)
                    train_spell_feature_values, train_spell_names = classifier.extract_features(train_set, train_users, USERS_DICT, features_combination)
                    with open('non_europe_test_spell_feature_values.pkl', 'wb') as f:
                        pickle.dump(test_spell_feature_values, f)
                    with open('non_europe_test_spell_names.pkl', 'wb') as f:
                        pickle.dump(test_spell_names, f)
                    with open('non_europe_train_spell_feature_values.pkl', 'wb') as f:
                        pickle.dump(train_spell_feature_values, f)
                    with open('non_europe_train_spell_names.pkl', 'wb') as f:
                        pickle.dump(train_spell_names, f)

            else:
                with open('non_europe_train_set.pkl', 'rb') as f:
                    train_set = pickle.load(f)
                with open('non_europe_train_labels.pkl', 'rb') as f:
                    train_labels = pickle.load(f)
                with open('non_europe_test_set.pkl', 'rb') as f:
                    test_set = pickle.load(f)
                with open('non_europe_test_labels.pkl', 'rb') as f:
                    test_labels = pickle.load(f)
                if SPELLING:
                    with open('non_europe_test_spell_feature_values.pkl', 'rb') as f:
                        test_spell_feature_values = pickle.load(f)
                    with open('non_europe_test_spell_names.pkl', 'rb') as f:
                        test_spell_names = pickle.load(f)
                    with open('non_europe_train_spell_feature_values.pkl', 'rb') as f:
                        train_spell_feature_values = pickle.load(f)
                    with open('non_europe_train_spell_names.pkl', 'rb') as f:
                        train_spell_names = pickle.load(f)




        with open(tokenizer_file_name, 'rb') as f:
            tokenizer = pickle.load(f)
        encoded_train_text_chunks = tokenizer.texts_to_sequences(train_set)
        encoded_test_text_chunks = tokenizer.texts_to_sequences(test_set)
        
        le = preprocessing.LabelEncoder()
        encoded_train_labels = le.fit_transform(train_labels)
        encoded_test_labels = le.transform(test_labels)

        if SPELLING:
            spell_X_train = Utils.normalize(np.matrix(train_spell_feature_values))
            spell_X_test = Utils.normalize(np.matrix(test_spell_feature_values))
        else:
            spell_X_train = None
            spell_X_test = None
        return encoded_train_text_chunks, encoded_train_labels, encoded_test_text_chunks, encoded_test_labels, number_of_classes, spell_X_train, spell_X_test
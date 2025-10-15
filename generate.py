import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import KDTree
import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm
from typing import List, Tuple

class ContextNeighborStorage:
    def __init__(self, sentences, model, tokenizer=None):
        self.sentences = sentences
        self.model = model
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.tokenizer = tokenizer

    def load_precomputed_embeddings(self) -> None:
        """
        Load pre_computed embeddings from 'lookup/normed_embeddings.pickle'. Saves some time and compute if you've already computed embeddings for a large corpus.

        Returns:
            None
        """
        filename = 'lookup/normed_embeddings.pickle'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                # Serialize and dump the object into the file
                self.normed_embeddings = pickle.load(file)

    def load_sentence_ids(self) -> None:
        """
        Load sentence ids from 'lookup/sentence_ids.pickle'. Mostly a helper function, really no point to call alone.

        Returns:
            None
        """
        filename = 'lookup/sentence_ids.pickle'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                # Deserialize and load the object from the file
                self.sentence_ids = pickle.load(file)
        else:
            print(f"File '{filename}' does not exist.")

    def load_token_ids(self) -> None:
        """
        Load token ids from 'lookup/token_ids.pickle'. Mostly a helper function, really no point to call alone.

        Returns:
            None
        """
        
        filename = 'lookup/token_ids.pickle'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                # Deserialize and load the object from the file
                self.token_ids = pickle.load(file)
        else:
            print(f"File '{filename}' does not exist.")

    def load_all_tokens(self) -> None:
        """
        Load all tokens tokenized from te corpus from 'lookup/all_tokens.pickle'. Mostly a helper function, really no point to call alone.

        Returns:
            None
        """
        filename = 'lookup/all_tokens.pickle'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                # Deserialize and load the object from the file
                self.all_tokens = pickle.load(file)
        else:
            print(f"File '{filename}' does not exist.")

    def load_sentences(self) -> None:
        """
        Load text sentences from 'lookup/setences.pickle'. It's kind of redundant; you should have the text corpus anyways.

        Returns:
            None
        """
        filename = 'lookup/sentences.pickle'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                # Deserialize and load the object from the file
                self.sentences = pickle.load(file)
        else:
            print(f"File '{filename}' does not exist.")

    def process_sentences(self) -> None:
        """
        Preprocess sentences and store all relevant artifacts into respective pickle files (I love that it's called pickling). This process includes tokenization, calculating embeddings from your desired AutoModel, normalzing those embeddings, and saving them.

        Returns:
            None
        """

        si_filename = 'lookup/sentence_ids.pickle'
        ti_filename = 'lookup/token_ids.pickle'
        at_filename = 'lookup/all_tokens.pickle'
        s_filename = 'lookup/sentences.pickle'
        filename = 'lookup/normed_embeddings.pickle'

        if os.path.exists(si_filename) and os.path.exists(ti_filename) and os.path.exists(at_filename) and os.path.exists(s_filename) and os.path.exists(filename):
            print("It looks like you've already processed a corpus.")
            option = input("Continue processing anyways? (y/n): ")
            if option.lower() != 'y':
                print("Skipping corpus processing.")
                return

        encoded_inputs = []
        with tqdm(total=len(self.sentences), desc="Tokenizing sentences") as pbar:
            for sentence in self.sentences:
                inputs_raw = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
                input_ids = inputs_raw['input_ids'].squeeze().tolist()  # Convert tensor to list
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
                inputs = {key: value.to(device) for key, value in inputs_raw.items()}
                with torch.no_grad():
                    results = self.model(**inputs)
                    encoded_inputs.append((tokens, results.last_hidden_state[0].cpu().numpy()))
                pbar.update()

        self.sentence_ids = []
        self.token_ids = []
        self.all_tokens = []
        all_embeddings = []
        for i, (toks, embs) in enumerate(tqdm(encoded_inputs, desc="appending toks and embeddings")):
            for j, (tok, emb) in enumerate(zip(toks, embs)):
                self.sentence_ids.append(i)
                self.token_ids.append(j)
                self.all_tokens.append(tok)
                all_embeddings.append(emb)
        all_embeddings = np.stack(all_embeddings)
        # we normalize embeddings, so that euclidian distance is equivalent to cosine distance
        self.normed_embeddings = (all_embeddings.T / (all_embeddings ** 2).sum(axis=1) ** 0.5).T

        # Check if the file already exists
        if not os.path.exists(si_filename):
            # Open the file in write-binary mode to create it if it doesn't exist
            if not os.path.exists('lookup/'):
                os.makedirs('lookup')
            with open(si_filename, 'wb') as file:
                # Serialize and dump the object into the file
                pickle.dump(self.sentence_ids, file)
            print(f"sentence ids saved to '{si_filename}'")
        else:
            print(f"File '{si_filename}' already exists. Skipping writing to avoid overwriting.")

        # Check if the file already exists
        if not os.path.exists(ti_filename):
            # Open the file in write-binary mode to create it if it doesn't exist
            with open(ti_filename, 'wb') as file:
                # Serialize and dump the object into the file
                pickle.dump(self.token_ids, file)
            print(f"token ids saved to '{ti_filename}'")
        else:
            print(f"File '{ti_filename}' already exists. Skipping writing to avoid overwriting.")

        # Check if the file already exists
        if not os.path.exists(at_filename):
            # Open the file in write-binary mode to create it if it doesn't exist
            with open(at_filename, 'wb') as file:
                # Serialize and dump the object into the file
                pickle.dump(self.all_tokens, file)
            print(f"all tokens saved to '{at_filename}'")
        else:
            print(f"File '{at_filename}' already exists. Skipping writing to avoid overwriting.")

        # Check if the file already exists
        if not os.path.exists(s_filename):
            # Open the file in write-binary mode to create it if it doesn't exist
            with open(s_filename, 'wb') as file:
                # Serialize and dump the object into the file
                pickle.dump(self.sentences, file)
            print(f"all tokens saved to '{s_filename}'")
        else:
            print(f"File '{s_filename}' already exists. Skipping writing to avoid overwriting.")


        # Check if the file already exists
        if not os.path.exists(filename):
            # Open the file in write-binary mode to create it if it doesn't exist
            with open(filename, 'wb') as file:
                # Serialize and dump the object into the file
                pickle.dump(self.normed_embeddings, file)
            print(f"Normed embeddings saved to '{filename}'")
        else:
            print(f"File '{filename}' already exists. Skipping writing to avoid overwriting.")

    def build_search_index(self) -> None:
        """
        Builds the KDTree from self.normed_embeddings

        Returns:
            None
        """
        self.indexer = KDTree(self.normed_embeddings)

    def query(
            self,
            query_sent : str,
            query_word : str,
            k :int = 10,
            filter_same_word : bool = True) -> Tuple[List[float], List[str], List[str]]:
        """
        Query the built KDTree with a desired word to find semantic neighbors for and a sentence to provide context.

        Args:
            query_sent (str): Sentence providing context for the target word.
            query_word (str): Target word.
            k (int): How many neighbors to search for
            filter_same_word (bool): Whether or not you want to filter out the same word (likely in slightly different contexts, but it will prevent return of the same token).

        Returns:
        Tuple[List[float], List[str], List[str]]: 
            - distances: List of floats representing distances of the neighbors from the target word.
            - neighbors: List of strings representing the found neighboring tokens.
            - contexts: List of strings representing contexts in which the semantic neighbors were found.
        """
        inputs_raw = tokenizer(query_sent, padding=True, truncation=True, return_tensors='pt')
        input_ids = inputs_raw['input_ids'].squeeze().tolist()  # Convert tensor to list
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        inputs = {key: value.to(device) for key, value in inputs_raw.items()}
        with torch.no_grad():
            results = self.model(**inputs)
            embs = (results.last_hidden_state[0].cpu().numpy())
            toks = tokens

        found = False
        for tok, emb in zip(toks, embs):
            if tok == query_word:
                found = True
                break
        if not found:
            raise ValueError('The query word {} is not a single token in sentence {}. You might want to try feeding in the first or last token of the target word.'.format(query_word, toks))
        emb = emb / sum(emb ** 2) ** 0.5
        initial_k = k
        di, idx = self.indexer.query(emb.reshape(1, -1), k=initial_k)
        distances = []
        neighbors = []
        contexts = []
        for i, index in enumerate(idx.ravel()):
            token = self.all_tokens[index]
            if filter_same_word and (query_word in token or token in query_word):
                continue
            distances.append(di.ravel()[i])
            neighbors.append(token)
            contexts.append(self.sentences[self.sentence_ids[index]])
            if len(distances) == k:
                break
        return distances, neighbors, contexts

    def direct_query(self,
        average_embedding : np.ndarray,
        word : str,
        k : int = 10,
        filter_same_word : bool = True):
        """
        Query the built KDTree with a desired word to find semantic neighbors for and a sentence to provide context.

        Args:
            average_embedding (np.ndarray): Averaged, non-normalized embedding representing a particular word from various contexts.
            word (str): Target word for display purposes.
            k (int): How many neighbors to search for
            filter_same_word (bool): Whether or not you want to filter out the same word (likely in slightly different contexts, but it will prevent return of the same token).

        Returns:
        Tuple[List[float], List[str], List[str]]: 
            - distances: List of floats representing distances of the neighbors from the target word.
            - neighbors: List of strings representing the found neighboring tokens.
            - contexts: List of strings representing contexts in which the semantic neighbors were found.
        """
        average_embedding = (average_embedding / sum(average_embedding ** 2) ** 0.5).T
        initial_k = k
        di, idx = self.indexer.query(average_embedding.reshape(1, -1), k=initial_k)
        distances = []
        neighbors = []
        contexts = []
        for i, index in enumerate(idx.ravel()):
            token = self.all_tokens[index]
            if filter_same_word and (word in token or token in word):
                continue
            distances.append(di.ravel()[i])
            neighbors.append(token)
            contexts.append(self.sentences[self.sentence_ids[index]])
            if len(distances) == k:
                break
        return distances, neighbors, contexts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "Semantic Pair Generator",
        description = "Given a set of sentences and target words and a corpus, computes embeddings for the words in context and searches the corpus for tokens similar in meaning (on the basis of distance in the embedding space)."
    )

    parser.add_argument("sentences_file", type=str)
    parser.add_argument("word_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--use_local_model", default="", type=str)
    parser.add_argument("--use_local_tokenizer", default="", type=str)

    args = parser.parse_args()

    assert os.path.exists(args.sentences_file), "Sentence file does not exist."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # load model

    if args.use_local_model != "":
        model = AutoModel.from_pretrained(args.use_local_model)
    else:
        model = AutoModel.from_pretrained(args.model_name)

    if args.use_local_tokenizer != "":
        tokenizer = AutoTokenizer.from_pretrained(args.use_local_tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = model.to(device)

    # open the sentence file
    with open(args.sentences_file, "r", encoding="utf-8") as f:
        data = [l.strip() for l in f.readlines()]

    # open the word file and parse
    # tsv, first column word to look for, second column word id (in case disambiguation is necessary), third column context
    word_dict = {}
    with open(args.word_file, "r", encoding="utf-8") as f:
        words = [l.strip().split('\t') for l in f.readlines()]
    for word in words:
        word_dict.setdefault(word[0], {})
        word_dict[word[0]].setdefault(word[1], [])
        word_dict[word[0]][word[1]].append(word[2])

    words_with_embeddings = {}
    # calculate embeddings for words
    for word, senses in word_dict.items():
        for sense, sentences in senses.items():
            for sentence in sentences:
                inputs_raw = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
                input_ids = inputs_raw['input_ids'].squeeze().tolist()
                word_id = tokenizer(word.strip().strip('"'), return_tensors='pt')['input_ids'].squeeze().tolist()[1:-1]
                assert len(word_id) == 1, f"{word_id} is made up of multiple tokens. Not implemented yet."
                word_id = word_id[0]
                index = input_ids.index(word_id)
                inputs = {key: value.to(device) for key, value in inputs_raw.items()}
                with torch.no_grad():
                    result = model(**inputs)
                words_with_embeddings.setdefault(word, {})
                words_with_embeddings[word].setdefault(sense, [])
                words_with_embeddings[word][sense].append(result.last_hidden_state[0][index].cpu().numpy())

    for word, senses in words_with_embeddings.items():
        for sense, embeddings in senses.items():
         words_with_embeddings[word][sense] = np.mean(np.vstack(embeddings), axis=0)

    storage = ContextNeighborStorage(sentences=data, model=model, tokenizer=tokenizer)
    storage.load_precomputed_embeddings()
    storage.load_sentence_ids()
    storage.load_sentences()
    storage.load_token_ids()
    storage.load_all_tokens()
    storage.process_sentences()
    storage.build_search_index()

    with open(args.output_file, "w+", encoding="utf-8") as f:
        f.write("Form\tDisambiguation\tFound Token\tCosine Similarity\tContext\n")
        for word, senses in words_with_embeddings.items():
            for sense, embedding in senses.items():
                distances, neighbors, contexts = storage.direct_query(
                    average_embedding=embedding, word=word, k=args.top_k)
                for d, w, c in zip(distances, neighbors, contexts):
                    f.write(f"{word}\t{sense}\t{w}\t{d}\t{c.split('\t')[1]}\n")



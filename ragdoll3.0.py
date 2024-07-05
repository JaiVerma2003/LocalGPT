import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

class TextAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Analyzer")

        self.text_box = ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.text_box.grid(row=0, column=0, padx=5, pady=5, columnspan=2)

        self.upload_button = tk.Button(root, text="Upload", command=self.upload_file)
        self.upload_button.grid(row=1, column=0, padx=5, pady=5, columnspan=2)

        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.grid(row=2, column=0, padx=5, pady=5, columnspan=2)

        self.create_analysis_buttons()

        self.common_input = tk.Text(self.root, wrap=tk.WORD, width=80, height=2)
        self.submit_button = tk.Button(self.root, text="Submit", command=self.perform_analysis)
        self.output_text = ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.sentiment_output_text = ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.summary_output_text = ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.name_output_text = ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.rephrase_output_text = ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)
        self.grammar_output_text = ScrolledText(self.root, wrap=tk.WORD, width=80, height=20)

        # Create common widgets initially but hide them
        self.common_input.grid(row=4, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.output_text.grid(row=6, column=0, padx=5, pady=5, columnspan=2)
        self.hide_common_widgets()

    def hide_common_widgets(self):
        self.common_input.grid_remove()
        self.submit_button.grid_remove()
        self.output_text.grid_remove()
        self.sentiment_output_text.grid_remove()
        self.summary_output_text.grid_remove()
        self.name_output_text.grid_remove()
        self.rephrase_output_text.grid_remove()
        self.grammar_output_text.grid_remove()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("PDF Files", "*.pdf"), ("Word Files", "*.docx;*.doc")])

        if file_path:
            # Hide the upload button after selecting a file
            self.upload_button.grid_forget()

            # Show additional buttons
            self.create_analysis_buttons()

            # Read and display the content of the file
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                text = str(pages)
            elif file_path.endswith((".doc", ".docx")):
                text = docx2txt.process(file_path)
            else:
                with open(file_path, 'r') as file:
                    text = file.read()
            
            self.display_text(text)

    def create_analysis_buttons(self):
        analysis_buttons = [
            ("QnA", self.show_qna_widgets),
            ("Summary", self.show_summary_widgets),
            ("Sentiment Analysis", self.show_sentiment_widgets),
            ("Grammar Analysis", self.show_grammar_widgets),
            ("Count", self.show_count_widgets),
            ("Dictionary", self.show_dictionary_widgets),
            ("Name Identification", self.show_name_widgets),
            ("Rephrase", self.show_rephrase_widgets)
        ]

        for index, (text, command) in enumerate(analysis_buttons):
            button = tk.Button(self.buttons_frame, text=text, command=command)
            button.grid(row=0, column=index, padx=5, pady=5)

    def show_qna_widgets(self):
        self.hide_common_widgets()
        self.common_input.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.grid(row=6, column=0, padx=5, pady=5, columnspan=2)
        self.output_text.grid(row=7, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.configure(command=self.perform_qna)

    def show_count_widgets(self):
        self.hide_common_widgets()
        self.common_input.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.grid(row=6, column=0, padx=5, pady=5, columnspan=2)
        self.output_text.grid(row=7, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.configure(command=self.perform_count)

    def show_dictionary_widgets(self):
        self.hide_common_widgets()
        self.common_input.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.grid(row=6, column=0, padx=5, pady=5, columnspan=2)
        self.output_text.grid(row=7, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.configure(command=self.perform_dictionary)

    def show_grammar_widgets(self):
        self.hide_common_widgets()
        self.submit_button.grid(row=4, column=0, padx=5, pady=5, columnspan=2)
        self.grammar_output_text.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.configure(command=self.perform_grammar_analysis)

    def show_sentiment_widgets(self):
        self.hide_common_widgets()
        self.submit_button.grid(row=4, column=0, padx=5, pady=5, columnspan=2)
        self.sentiment_output_text.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.configure(command=self.perform_sentiment_analysis)

    def show_summary_widgets(self):
        self.hide_common_widgets()
        self.submit_button.grid(row=4, column=0, padx=5, pady=5, columnspan=2)
        self.summary_output_text.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.configure(command=self.perform_summary)

    def show_name_widgets(self):
        self.hide_common_widgets()
        self.submit_button.grid(row=4, column=0, padx=5, pady=5, columnspan=2)
        self.name_output_text.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.configure(command=self.perform_name_identification)

    def show_rephrase_widgets(self):
        self.hide_common_widgets()
        self.submit_button.grid(row=4, column=0, padx=5, pady=5, columnspan=2)
        self.rephrase_output_text.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.configure(command=self.perform_rephrase)

    '''def show_common_widgets(self):
        self.common_input.grid(row=5, column=0, padx=5, pady=5, columnspan=2)
        self.submit_button.grid(row=6, column=0, padx=5, pady=5, columnspan=2)
        self.output_text.grid(row=7, column=0, padx=5, pady=5, columnspan=2)'''

    def perform_qna(self):
        print("qna")
        user_input = self.text_box.get(1.0, tk.END)
        query = self.common_input.get(1.0, tk.END)
        result = final_result(user_input, query)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "User's Question:\n" + result)
        print("qnaend")

    def perform_summary(self):
        print("summary")
        user_input = self.text_box.get(1.0, tk.END)
        query = "Summarise the following data"
        result = final_result(user_input, query)
        print(result)
        self.summary_output_text.delete(1.0, tk.END)
        self.summary_output_text.insert(tk.END, "Summary Output:\n" + result)

    def perform_sentiment_analysis(self):
        print("sentiment")
        user_input = self.text_box.get(1.0, tk.END)
        query = "Perform a sentiment analysis on the following data and provide the relevant output"
        result = final_result(user_input, query)
        self.sentiment_output_text.delete(1.0, tk.END)
        self.sentiment_output_text.insert(tk.END, "Sentiment Analysis Output:\n" + result)

    def perform_grammar_analysis(self):
        print("grammar")
        user_input = self.text_box.get(1.0, tk.END)
        query = "Rewrite the following data with proper grammar and punctuation without changing number of sentences or format of the data"
        result = final_result(user_input, query)
        print(result)
        self.grammar_output_text.delete(1.0, tk.END)
        self.grammar_output_text.insert(tk.END, "Grammar Analysis Output:\n" + result)

    def perform_count(self):
        print("count")
        user_input = self.text_box.get(1.0, tk.END)
        query_format = "How many number of times does '{}' appear in the following data?"
        word = self.common_input.get(1.0, tk.END)
        query = query_format.format(word)
        print(query)
        result = final_result(user_input, query)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "User's Count:\n" + result)

    def perform_dictionary(self):
        user_input = self.common_input.get(1.0, tk.END)
        query_format = "What is the meaning of '{}'"
        word = self.common_input.get(1.0, tk.END)
        query = query_format.format(word)
        result = final_result(user_input, query)
        print(result)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "User's Dictionary:\n" + result)

    def perform_name_identification(self):
        user_input = self.text_box.get(1.0, tk.END)
        query = "Identify all the different names present in the following data"
        result = final_result(user_input, query)
        self.name_output_text.delete(1.0, tk.END)
        self.name_output_text.insert(tk.END, "Identified Name:\n" + result)

    def perform_rephrase(self):
        user_input = self.text_box.get(1.0, tk.END)
        query = "Rephrase the following data in approxmately the same number of words"
        result = final_result(user_input, query)
        print(result)
        self.rephrase_output_text.delete(1.0, tk.END)
        self.rephrase_output_text.insert(tk.END, "Rephrased Text:\n" + result)

    def perform_analysis(self):
        '''current_button = [widget for widget in self.buttons_frame.winfo_children() if widget.cget("state") != "disabled"]
        if current_button:
            current_button[0].invoke()'''
        if analysis_type == "QnA":
            self.perform_qna()
        elif analysis_type == "Count":
            self.perform_count()
        elif analysis_type == "Dictionary":
            self.perform_dictionary()

    def display_text(self, content):
        self.text_box.delete(1.0, tk.END)
        self.text_box.insert(tk.END, content)

import os
import pandas as pd
import matplotlib.pyplot as plt
#from docx import Document
import docx2txt
from transformers import GPT2TokenizerFast
from huggingface_hub import hf_hub_download
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from datetime import datetime

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def load_model(device_type, model_id, model_basename=None):
    model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
    print(model_path)
    max_ctx_size = 2048
    kwargs = {
        "model_path": model_path,
        "n_ctx": max_ctx_size,
        "max_tokens": max_ctx_size,
    }
    kwargs["n_gpu_layers"] = 1000
    kwargs["n_batch"] = max_ctx_size
    return LlamaCpp(**kwargs)

'''llm = load_model(device_type = "cuda", model_id="TheBloke/Llama-2-7B-Chat-GGML", model_basename="llama-2-7b-chat.ggmlv3.q4_0.bin")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L12-v2')
chain = load_qa_chain(llm, chain_type = "stuff")'''

file_path = ""

def current_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("Current Time:", formatted_time)

def final_result(text, query):
    print("calculation starts at: ", end = "")
    current_time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 256,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )

    chunks = text_splitter.create_documents([text])
    db = FAISS.from_documents(chunks, embeddings)

    docs = db.similarity_search(query)
    print("goin in to get the result")
    result = chain.run(input_documents = docs, question = query)
    print("calculation ends at: ", end = "")
    chunks.clear()
    docs.clear()
    current_time()
    return result

if __name__ == "__main__":

    current_time()
    
    llm = load_model(device_type = "cuda", model_id="TheBloke/Llama-2-7B-Chat-GGML", model_basename="llama-2-7b-chat.ggmlv3.q4_0.bin")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L12-v2')
    chain = load_qa_chain(llm, chain_type = "stuff")
    
    root = tk.Tk()
    app = TextAnalyzerApp(root)
    root.mainloop()


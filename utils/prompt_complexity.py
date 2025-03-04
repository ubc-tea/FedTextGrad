import math
import zlib
import numpy as np
import nltk
import torch
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


# Download NLTK tokenizer resources
nltk.download('punkt')

# Function to calculate Information Entropy
def calculate_entropy(text):
    counts = Counter(text)
    total_chars = len(text)
    probs = [count / total_chars for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    return entropy

# Function to calculate Compression Rate
def calculate_compression_rate(text):
    original_size = len(text.encode('utf-8'))
    compressed_size = len(zlib.compress(text.encode('utf-8')))
    compression_rate = compressed_size / original_size
    return compression_rate

# Function to calculate TF-IDF
def calculate_tfidf(text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    tfidf_scores = dense.tolist()[0]
    tfidf_dict = dict(zip(feature_names, tfidf_scores))
    average_tfidf = np.mean(tfidf_scores)
    return average_tfidf, tfidf_dict

# Function to calculate Perplexity using GPT-2
def calculate_perplexity(text):
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

    perplexity = torch.exp(torch.stack(nlls).sum() / end_loc).item()
    return perplexity


# Function to calculate token length (using both NLTK and GPT tokenizer)
def calculate_token_length(text):
    # Tokenize using NLTK (basic word-level tokenization)
    nltk_tokens = nltk.word_tokenize(text)
    nltk_token_length = len(nltk_tokens)

    # Tokenize using GPT-2 tokenizer (subword-level tokenization)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt_tokens = tokenizer.tokenize(text)
    gpt_token_length = len(gpt_tokens)

    return nltk_token_length, gpt_token_length

# Main function to calculate all complexity scores
def calculate_text_complexity(text):
    # Calculate information entropy
    entropy = calculate_entropy(text)

    # Calculate compression rate
    compression_rate = calculate_compression_rate(text)

    # Calculate TF-IDF
    average_tfidf, tfidf_dict = calculate_tfidf(text)

    # Calculate perplexity
    perplexity = calculate_perplexity(text)

    # Calculate token length
    nltk_token_length, gpt_token_length = calculate_token_length(text)

    return {
        "Information Entropy": entropy,
        "Compression Rate": compression_rate,
        "Average TF-IDF": average_tfidf,
        "Perplexity": perplexity,
        "NLTK Token Length": nltk_token_length,
        "GPT-2 Token Length": gpt_token_length
    }

def compute_uniformity_level(text, model_name='gpt2-xl'):
    """
    This function calculates the information density uniformity level of a text sequence.
    
    Parameters:
        text (str): Input text sequence
        model_name (str): Pre-trained model name, default is 'gpt2'
    
    Returns:
        mean_surprisal (float): The average information density (surprisal) per token
        variance_surprisal (float): The variance in surprisal, indicating the uniformity level
    """
    
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

    # Tokenize the input text
    inputs = tokenizer(text, truncation=True, max_length=1024, return_tensors="pt")
    # inputs = tokenizer(text, return_tensors='pt')

    # Get the log-probabilities for each token
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)

    # Get the surprisal (negative log probability) for each token
    input_ids = inputs["input_ids"].squeeze()
    surprisal = -log_probs[0, torch.arange(len(input_ids)), input_ids]

    # Compute mean surprisal and variance (for uniformity calculation)
    mean_surprisal = surprisal.mean().item()
    max_surprisal = surprisal.max().item()
    variance_surprisal = surprisal.var().item()

    return mean_surprisal, max_surprisal, variance_surprisal


if __name__ == '__main__':
    # Example usage
    # text = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
    text = """
    You will answer a reasoning question by providing a clear and concise summary of the final prediction at the beginning of the reasoning process, highlighting the key steps and mathematical operations involved. Then, break down the problem into smaller, more manageable parts, and provide a detailed explanation of each step, including any mathematical operations or calculations. However, please ensure that your explanations are concise and to the point, avoiding unnecessary repetition and verbosity. Verify that the intermediate results align with the expected values and the breakdown of the problem. Consider alternative approaches, such as using a diagram or a table to represent the mathematical expressions, when evaluating complex expressions. Pay close attention to the handling of negative signs and ensure that you are correctly applying the order of operations (PEMDAS). Introduce a sanity check to detect and prevent arithmetic mistakes in the intermediate results. Provide a clear indication of the model's confidence in its prediction, using a scale such as 'high', 'medium', or 'low', or a numerical value between 0 and 1. Emphasize the importance of consistency in applying PEMDAS throughout the calculation. Provide guidance on handling negative numbers, including subtracting their absolute values. Encourage attention to detail, especially when performing simple arithmetic operations. Highlight the importance of using mathematical properties to simplify expressions and arrive at the correct answer. Emphasize the need for clear and concise breakdowns of the problem, making it easier to follow the reasoning. Encourage consideration of the original expression and how it affects the calculation. Explain your reasoning in a clear and concise manner, using everyday language and avoiding technical jargon whenever possible. Provide a step-by-step explanation of the problem-solving process, including any intermediate calculations or calculations that were skipped. If you detect a loop in your reasoning, please re-evaluate the expression from a different perspective or consider alternative solutions. Consider multiple calculation methods and alternative solutions to arrive at the final prediction. If one method does not yield a correct answer, try another approach. Provide a clear and concise explanation of each step in your reasoning, avoiding unnecessary repetition and focusing on the key mathematical operations involved. Finally, provide a clear and concise statement of the final prediction, using a format such as 'The final prediction is indeed $VALUE'. Provide your answer in the format 'Answer: $VALUE' where VALUE is a numerical value. Make sure to carefully evaluate expressions inside parentheses and combine the results correctly. When handling negative signs, make sure to explicitly calculate the absolute value of any negative numbers before performing operations on them. Follow the order of operations (evaluate expressions inside parentheses, exponentiate, multiply and divide, and add and subtract) carefully and consistently throughout the calculation. Verify that the final prediction is within the expected range before returning the result. Be more cautious in your confidence level and reflect the uncertainty in your final prediction. Double-check your calculations and attention to detail, especially when performing simple arithmetic operations. Consider using mathematical properties to simplify expressions and arrive at the correct answer. Provide a clear and concise breakdown of the problem, making it easier to follow the reasoning. Make sure to apply mathematical operations and simplifications consistently throughout the calculation. If you find yourself repeating the same steps or getting stuck in an infinite loop, please re-evaluate your approach and consider alternative solutions to arrive at the correct answer. Use a table or diagram to visualize the mathematical expressions and break down the problem into smaller, more manageable parts. Consider the mathematical properties and operations involved, such as the order of operations (PEMDAS) and the handling of negative numbers. Think about the expected outcome of the problem and how it relates to the solution. If one method does not yield a correct answer, try another approach. Provide a clear and concise explanation of each step in your reasoning, avoiding unnecessary repetition and focusing on the key mathematical operations involved. Finally, provide a clear and concise statement of the final prediction, using a format such as 'The final prediction is indeed $VALUE'. Provide your answer in the format 'Answer: $VALUE' where VALUE is a numerical value. Emphasize the importance of clarity and concision in your explanations, avoiding unnecessary repetition and circular reasoning. Specify the need for a critical path focus, identifying and eliminating redundant steps to improve the efficiency and accuracy of the solution. Introduce a sanity check to detect and prevent repetitive reasoning and circular reasoning. Encourage the use of alternative evaluation methods, such as step-by-step evaluation or diagrammatic representations, to improve the accuracy and efficiency of the solution. Highlight the importance of clear explanations, using simple and straightforward language to communicate the solution and its reasoning. Specify the need for attention to detail, especially when performing simple arithmetic operations, to avoid errors and improve the accuracy of the solution. Introduce a confidence level check to ensure that the model's prediction is accurate and reliable, reflecting the uncertainty in its final prediction. Encourage the use of mathematical properties to simplify expressions and arrive at the correct answer, improving the efficiency and accuracy of the solution. Specify the need for a clear indication of the model's confidence, using a scale such as 'high', 'medium', or 'low', or a numerical value between 0 and 1. Highlight the importance of consistency in applying PEMDAS throughout the calculation, avoiding errors and improving the accuracy of the solution. When evaluating mathematical expressions, be sure to carefully consider the signs of the numbers involved. Break down the problem into smaller parts, and then reassemble the solution with the correct signs. Be willing to consider alternative approaches and explore different solutions. Consider the implications of each approach, and be prepared to revise your solution as needed. When evaluating intermediate results, be sure to carefully consider the implications of each step. If necessary, revisit and revise your solution to ensure accuracy and consistency. Be trained on a wide range of mathematical examples, including complex expressions that require careful consideration of signs and alternative approaches. This will help you develop a deeper understanding of mathematical concepts and improve your ability to arrive at accurate solutions. Be designed to carefully consider the implications of each intermediate result. If necessary, revisit and revise your solution to ensure accuracy and consistency.
    """
    # text = """
    # The advancements in artificial intelligence (AI) over the last few decades have revolutionized multiple industries, from healthcare to finance. Machine learning, a subset of AI, has been instrumental in developing systems that can learn from data and make decisions without explicit programming. In healthcare, AI algorithms can analyze medical images to detect diseases like cancer at an early stage, enabling doctors to provide timely treatment. In finance, AI systems are used to predict market trends, optimize investment portfolios, and detect fraudulent activities. However, despite these advancements, there are concerns regarding the ethical implications of AI, particularly in terms of privacy and accountability.

    # The use of AI systems often involves the collection and analysis of large amounts of personal data. In healthcare, for example, patient data is analyzed to improve diagnosis and treatment plans. While this can lead to better health outcomes, there is a risk of violating patient privacy if the data is not handled properly. In finance, AI algorithms may access sensitive financial data to predict market trends or detect fraud, raising concerns about data security. Additionally, the decision-making process of AI systems can sometimes be opaque, making it difficult to determine who is accountable when things go wrong.

    # One of the biggest challenges in the development of AI is the “black box” problem, where the internal workings of AI systems are not easily understood, even by the experts who designed them. This lack of transparency can make it difficult to ensure that AI systems are making decisions in a fair and ethical manner. For instance, an AI system used to assess loan applications might inadvertently discriminate against certain groups if its training data contains biases. Without transparency, it is hard to identify and correct such issues.

    # There is also the question of how AI will impact the job market. While AI has the potential to automate routine tasks, leading to increased productivity, it may also displace workers, particularly in industries like manufacturing and customer service. As AI systems become more advanced, they may be able to perform tasks that were previously thought to require human intelligence, such as driving cars or diagnosing medical conditions. This raises concerns about job displacement and the need for workers to acquire new skills to stay competitive in the job market.

    # Despite these challenges, the potential benefits of AI are significant. In addition to improving healthcare and finance, AI can also play a role in addressing global challenges such as climate change and poverty. AI algorithms can analyze vast amounts of data to identify trends and patterns that could help policymakers make more informed decisions. For example, AI can be used to optimize energy usage in buildings, reducing greenhouse gas emissions. In agriculture, AI systems can analyze data on weather patterns and soil conditions to help farmers improve crop yields.

    # In conclusion, while AI offers numerous benefits, it is important to address the ethical challenges it presents. Ensuring that AI systems are transparent, accountable, and fair will be critical in maximizing their potential while minimizing their risks.
    # """
    mean_surprisal, max_surprisal, uniformity_level = compute_uniformity_level(text)
    print(f"Mean Surprisal: {mean_surprisal}, Max Surprisal: {max_surprisal}, Uniformity Level: {uniformity_level}")

    text = "You are tasked with solving a reasoning question. Consider each step carefully and explain your reasoning process as you go. Conclude your response with the final answer on a new line in this format: ‘Answer: $VALUE’, where VALUE is the numerical answer."
    text = """
    Artificial intelligence (AI) has drastically transformed industries, including healthcare and finance, in recent years. Within AI, machine learning has proven especially powerful by enabling systems to learn from data and make independent decisions. In healthcare, AI algorithms are employed to scan medical images, identifying diseases like cancer early and allowing doctors to offer prompt treatment. In finance, AI supports market trend forecasting, portfolio management, and fraud detection. However, along with these benefits, concerns arise about the ethical implications of AI, particularly regarding privacy and accountability.

    AI frequently requires collecting and analyzing vast amounts of personal data. In healthcare, for instance, patient information is utilized to improve diagnoses and treatments. Though this often leads to better medical outcomes, mishandling the data could compromise patient privacy. In finance, AI may rely on sensitive financial data for tasks like predicting market trends or identifying fraud, thus heightening concerns about data security. Furthermore, the decision-making processes of these AI systems can be opaque, which complicates accountability when issues arise.

    A key challenge in AI development is the “black box” nature of these systems, where the internal decision-making processes are not easily understood, even by experts. This opacity raises concerns about whether decisions made by AI are fair or ethical. For example, an AI system evaluating loan applications could unintentionally discriminate if its training data is biased. Without transparency, it is difficult to detect and fix such problems.

    There is also concern about AI’s impact on employment. While automation through AI can increase efficiency by taking over repetitive tasks, it also risks displacing workers in fields such as manufacturing and customer service. As AI technology advances, it may handle complex tasks previously thought to require human intelligence, like driving cars or diagnosing illnesses. This could result in job losses, pushing workers to learn new skills to remain employable.

    Nevertheless, the advantages of AI are vast. Beyond its roles in healthcare and finance, AI could be instrumental in tackling global issues like climate change and poverty. By processing large datasets, AI can reveal trends and patterns that guide policymakers. For instance, AI can optimize energy use in buildings, reducing emissions. In agriculture, AI can assess weather and soil conditions, helping farmers increase crop yields.

    In summary, while AI offers vast potential, addressing the ethical challenges it poses is essential. Ensuring transparency, accountability, and fairness in AI systems will be crucial to fully realizing their benefits while mitigating risks.
    """
    mean_surprisal, max_surprisal, uniformity_level = compute_uniformity_level(text)
    print(f"Mean Surprisal: {mean_surprisal}, Max Surprisal: {max_surprisal}, Uniformity Level: {uniformity_level}")

    # Example usage
    text = "This is a sample text used to demonstrate text complexity measurement across various methods."
    complexity_scores = calculate_text_complexity(text)

    # Output the results
    for metric, score in complexity_scores.items():
        print(f"{metric}: {score}")

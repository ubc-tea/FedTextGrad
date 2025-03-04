from string import Template

SUMMARIZATION_TEMPLATE = Template("""
Merge the following list of prompts into a single, cohesive prompt while preserving all original information. Ensure that the final instruction remains unchanged and is placed as the last sentence. Provide only the merged prompt.

List of Prompts to Merge:
$prompt
"""
)

UID_TEMPLATE = Template("""
Merge the following list of prompts into a single, cohesive prompt while preserving all original information. Apply Uniform Information Density Principles. Ensure that the final instruction remains unchanged and is placed as the last sentence. Provide only the merged prompt.

List of Prompts to Merge:
$prompt
""")

FORMATTING_INSTRUCTION = "The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."

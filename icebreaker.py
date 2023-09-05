from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin import scrape_linkedin_profile

# information = """
#     Nikola Tesla (/ˈtɛslə/; Serbian Cyrillic: Никола Тесла,[2] pronounced [nǐkola têsla];[a] 10 July [O.S. 28 June] 1856 – 7 January 1943) was a Serbian-American[5][6] inventor, electrical engineer, mechanical engineer, and futurist best known for his contributions to the design of the modern alternating current (AC) electricity supply system.[7]
#
#     Born and raised in the Austrian Empire, Tesla studied engineering and physics in the 1870s without receiving a degree, gaining practical experience in the early 1880s working in telephony and at Continental Edison in the new electric power industry. In 1884 he emigrated to the United States, where he became a naturalized citizen. He worked for a short time at the Edison Machine Works in New York City before he struck out on his own. With the help of partners to finance and market his ideas, Tesla set up laboratories and companies in New York to develop a range of electrical and mechanical devices. His AC induction motor and related polyphase AC patents, licensed by Westinghouse Electric in 1888, earned him a considerable amount of money and became the cornerstone of the polyphase system which that company eventually marketed.
#
#     Attempting to develop inventions he could patent and market, Tesla conducted a range of experiments with mechanical oscillators/generators, electrical discharge tubes, and early X-ray imaging. He also built a wirelessly controlled boat, one of the first ever exhibited. Tesla became well known as an inventor and demonstrated his achievements to celebrities and wealthy patrons at his lab, and was noted for his showmanship at public lectures. Throughout the 1890s, Tesla pursued his ideas for wireless lighting and worldwide wireless electric power distribution in his high-voltage, high-frequency power experiments in New York and Colorado Springs. In 1893, he made pronouncements on the possibility of wireless communication with his devices. Tesla tried to put these ideas to practical use in his unfinished Wardenclyffe Tower project, an intercontinental wireless communication and power transmitter, but ran out of funding before he could complete it.
#
#     After Wardenclyffe, Tesla experimented with a series of inventions in the 1910s and 1920s with varying degrees of success. Having spent most of his money, Tesla lived in a series of New York hotels, leaving behind unpaid bills. He died in New York City in January 1943.[8] Tesla's work fell into relative obscurity following his death, until 1960, when the General Conference on Weights and Measures named the International System of Units (SI) measurement of magnetic flux density the tesla in his honor. There has been a resurgence in popular interest in Tesla since the 1990s.[9]
# """
#
if __name__ == "__main__":
    print("hello there")

    linkedin_profile_url = linkedin_lookup_agent(name="Lucky Phan software")

    summary_template = """
        given the LinkedIn information {information} about a person, I want you to create:
        1. a short summary
        2. two interesting facts about them
    """
    #
    #     # input variables are strings/keys that will be populated with in list format
    #     # template is the text before injected with variables
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # temperature determines the creativeness of the model
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # chain allows it to run on the llm with the prompt template
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    print(chain.run(information=linkedin_data))

    # the input var is named information in the prompt template,  when created chain instance and provided prompt, when
    # the chain runs, the keyword arg information will be valid

    # print(linkedin_data)

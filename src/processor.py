import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List


logger = logging.getLogger(__name__)


class FirmIntelligence(BaseModel):
    firm_name: str = Field(description="Name of the firm")
    motto: str = Field(description="Company motto or tagline")
    hiring_keywords: List[str] = Field(description="List of 5 adjectives describing their ideal candidate")
    firm_tone: str = Field(description="Tone: 'Formal', 'Aggressive', 'Modern', or 'Community'")
    recent_wins: List[str] = Field(description="Summary of recent cases or deals found in text")
    lifestyle_summary: str = Field(description="Summary of work-life balance and culture")
    sector_focus: List[str] = Field(description="Top 3 industries they serve")


class DataProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=FirmIntelligence)

    def extract_intelligence(self, raw_text):
        prompt = PromptTemplate(
            template="""
            You are a Career Consultant analyzing a London Law Firm.
            Extract the following resume-tailoring data from the raw text below.
            If data is missing, make an educated guess based on the text tone.

            {format_instructions}

            RAW TEXT FROM WEBSITE:
            {text}
            """,
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

        chain = prompt | self.llm | self.parser

        try:
            return chain.invoke({"text": raw_text[:15000]})
        except Exception as e:
            logger.error("LLM Extraction Error: %s", e)
            return None

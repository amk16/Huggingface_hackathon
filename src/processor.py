import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional


logger = logging.getLogger(__name__)


class FirmIntelligence(BaseModel):
    firm_name: str = Field(description="Name of the firm")
    motto: str = Field(description="Company motto or tagline")
    hiring_keywords: List[str] = Field(description="List of 5 adjectives describing their ideal candidate")
    firm_tone: str = Field(description="Tone: 'Formal', 'Aggressive', 'Modern', or 'Community'")
    recent_wins: List[str] = Field(description="Summary of recent cases or deals found in text")
    lifestyle_summary: str = Field(description="Summary of work-life balance and culture")
    sector_focus: List[str] = Field(description="Top 3 industries they serve")


class CareerOpening(BaseModel):
    title: str = Field(default="", description="Role title or program name")
    practice_area: str = Field(default="", description="Practice area or business unit")
    location: str = Field(default="", description="Primary office or location mentioned")
    experience_level: str = Field(default="", description="Student, trainee, associate, etc.")
    application_deadline: str = Field(default="", description="Deadline text if provided")
    application_link: str = Field(default="", description="Direct application link if available")
    notes: str = Field(default="", description="Any other useful detail such as contract length")


class CareerInsights(BaseModel):
    hiring_focus: str = Field(default="", description="Summary of what the firm is recruiting for right now")
    application_channels: List[str] = Field(default_factory=list, description="Where or how to apply")
    benefits: List[str] = Field(default_factory=list, description="Key perks or benefits highlighted")
    interview_process: str = Field(default="", description="How the interview/assessment process works")
    candidate_tips: List[str] = Field(default_factory=list, description="Advice or qualities the firm emphasizes")
    current_openings: List[CareerOpening] = Field(default_factory=list, description="Specific openings found on the career page")


class DataProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.parser = JsonOutputParser(pydantic_object=FirmIntelligence)
        self.career_parser = JsonOutputParser(pydantic_object=CareerInsights)
        self.career_prompt = PromptTemplate(
            template="""
            You are a Career Consultant summarizing what applicants should know about this firm.

            Follow the requested JSON structure and only include items that appear in the text.

            {format_instructions}

            CAREER PAGE TEXT:
            {text}
            """,
            input_variables=["text"],
            partial_variables={"format_instructions": self.career_parser.get_format_instructions()},
        )

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

    def extract_career_insights(self, career_sections: List[dict], fallback_text: str) -> Optional[CareerInsights]:
        """Extract structured information directly from the firm's career pages."""
        if not career_sections and not fallback_text:
            return None

        if career_sections:
            snippets = []
            for section in career_sections[:6]:
                text = section.get("text", "")
                if not text:
                    continue
                snippets.append(f"URL: {section.get('url', '')}\n{text}")
            source_text = "\n\n".join(snippets)
        else:
            source_text = fallback_text

        if not source_text:
            return None

        chain = self.career_prompt | self.llm | self.career_parser

        try:
            return chain.invoke({"text": source_text[:15000]})
        except Exception as e:
            logger.error("Career insights extraction error: %s", e)
            return None
    
    def extract_job_keywords_tone(self, jobs_data: list):
        """
        Extract keywords and tone from job listings to add as metric.
        
        Args:
            jobs_data: List of job dictionaries
            
        Returns:
            Dictionary with keywords and tone extracted from job descriptions
        """
        if not jobs_data:
            return None
        
        # Combine all job titles and summaries
        job_text = "\n\n".join([
            f"Title: {job.get('title', '')}\nSummary: {job.get('summary', '')}"
            for job in jobs_data[:20]  # Limit to first 20 jobs
        ])
        
        prompt = PromptTemplate(
            template="""
            Analyze the following job listings and extract:
            1. Common keywords/terms that appear across these job postings
            2. The overall tone/approach to hiring (e.g., "Formal", "Modern", "Aggressive", "Community-focused")
            3. Key skills or qualifications they're seeking
            
            Return a JSON object with:
            - keywords: List of 5-10 key terms/phrases
            - tone: One word describing hiring approach
            - skills: List of 3-5 key skills/qualifications mentioned
            
            JOB LISTINGS:
            {text}
            """,
            input_variables=["text"],
        )
        
        try:
            # Use a simple JSON output parser for this
            from langchain_core.output_parsers import StrOutputParser
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"text": job_text[:10000]})
            
            # Try to parse as JSON
            import json
            try:
                # The LLM might return markdown code blocks, so clean it up
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()
                
                parsed = json.loads(result)
                return parsed
            except json.JSONDecodeError:
                # Fallback: return basic structure
                logger.warning("Could not parse job keywords as JSON, using fallback")
                return {
                    "keywords": ["legal", "law", "lawyer", "solicitor", "legal professional"],
                    "tone": "Professional",
                    "skills": ["legal expertise", "communication", "analytical skills"]
                }
        except Exception as e:
            logger.error(f"Error extracting job keywords/tone: {e}")
            return None
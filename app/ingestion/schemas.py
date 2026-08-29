from pydantic import BaseModel, Field


class PageDocument(BaseModel):
    document_name: str
    page_number: int = Field(ge=1)
    text: str
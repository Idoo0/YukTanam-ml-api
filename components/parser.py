from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

class PlantDisease(BaseModel):
    nama_tanaman: str = Field(description="Nama tanaman")
    nama_penyakit: str = Field(description="Nama penyakit yang menyerang tanaman")
    deskripsi_penyakit: str = Field(description="Penjelasan mengenai penyakit pada tanaman")
    penanganan_penyakit: List[str] = Field(description="Cara menangani penyakit pada tanaman")
    pencegahan_penyakit: List[str] = Field(description="Cara mencegah penyakit pada tanaman")
    point: int = Field(description="Point yang didapatkan berdasarkan kondisi tanaman")

def getParserComponent():
    parser = JsonOutputParser(pydantic_object=PlantDisease)
    format_instructions = parser.get_format_instructions()
     
    return {
      "parser": parser,
      "format_instructions": format_instructions
    }
"""Parser for html, docx, pdf."""

import re
from io import StringIO

from bs4 import BeautifulSoup
from docx import Document
from fire import Fire
from pdfminer3.converter import TextConverter
from pdfminer3.layout import LAParams
from pdfminer3.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer3.pdfpage import PDFPage


class Parser:
    def __init__(self):
        self.mapper = {
            "html": self._html,
            "doc": self._docx,
            "docx": self._docx,
            "pdf": self._pdf,
        }

    def parse(self, path, ext="html"):
        return self.mapper[ext](path)

    def _docx(self, file_path: str) -> str:
        """Load a docx file and split document into pages.
        Args:
            file_path: path to docx file.
        Returns:
            list of pages.
        """
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)

    def _html(self, path: str) -> str:
        """Split html content to pages.
        Args:
            raw_html: html content
        Returns:
            list of pages
        """
        with open(path) as file:
            raw_html = file.read()
        soup = BeautifulSoup(raw_html, features="lxml")

        # TODO: move to the config
        for div in soup.find_all(["header", "head", "nav", "footer"]):
            div.decompose()

        for div in soup.find_all(
            class_=["cookie-compliance__inner", "menu-selector", "page-search"]
        ):
            div.extract()

        for div in soup.find_all(id=["nav", "menuwrapper"]):
            div.extract()

        # replace non-breaking space
        return re.sub(r"\n+", "\n", soup.get_text())

    def _pdf(self, path: str) -> str:
        """Load a PDF file and split document to pages.
        Args:
            pdf: PDF file or path to file
        Returns:
            list of pages.
        """
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        device = TextConverter(rsrcmgr, retstr, codec="utf-8", laparams=LAParams())
        list_of_pages = []
        with open(path, "rb") as pdf_file:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            password = ""
            maxpages = 0
            caching = True
            pagenos = set()

            for page in PDFPage.get_pages(
                pdf_file,
                pagenos,
                maxpages=maxpages,
                password=password,
                caching=caching,
                check_extractable=True,
            ):
                read_position = retstr.tell()
                interpreter.process_page(page)
                retstr.seek(read_position, 0)
                page_text = retstr.read()
                list_of_pages.append(page_text)
        device.close()
        retstr.close()
        return "\n".join(list_of_pages)


if __name__ == "__main__":
    Fire(Parser)
import pytest
from pyempfin.htmloutput import HtmlOutput

def test_html_insert():
    htmlout = HtmlOutput(title='My Title')

    print(htmlout.fullhtml)
import datetime

class HtmlOutput:

    def __init__(self,
                 title: str='',
                 title_to_h1: bool=True,
                 insert_date: bool=True):
        """Create a new HTML file with specific title"""
        # HTML
        self._fullhtml = '<!DOCTYPE html>\n<html>\n<head>\n' + \
                         f'<meta charset="utf-8">\n<title>{title}</title>\n' + \
                         '</head>\n<body>\n</body>\n</html>'
        # Position of end-of-file
        self._endptr = self._fullhtml.find('</body>')
        # Add Heading 1
        if title_to_h1:
            self.insert_tag(title, 'h1')
        # Add date
        if insert_date:
            self.insert_str(
                'Date Created: ' +
                datetime.datetime.now().strftime('%Y-%m-%d (%a) %H:%M:%S') + '\n'
            )


    def _refresh_endptr(self):
        self._endptr = self._fullhtml.find('</body>')


    def insert_str(self, text: str):
        self._fullhtml = self._fullhtml[:self._endptr] + text + \
                         self._fullhtml[self._endptr:]
        self._refresh_endptr()


    def insert_tag(self, content: str, tag: str):
        self.insert_str(
            f'<{tag}>' + content + f'</{tag}>\n'
        )


    def save(self, filepath):
        with open(filepath, 'w') as f:
            f.write(self._fullhtml)


    def print(self):
        print(self._fullhtml)


    def __repr__(self):
        return self._fullhtml


    def __str__(self):
        return self._fullhtml


    @property
    def fullhtml(self):
        return self._fullhtml


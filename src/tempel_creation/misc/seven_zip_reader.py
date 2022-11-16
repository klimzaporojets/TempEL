import _io
import io
from typing import List, Dict, Any, Optional

import py7zlib
import py7zr
from py7zr.compressor import SevenZipDecompressor


class SevenZFileError(py7zlib.ArchiveError):
    pass


class SevenZFile(object):
    @classmethod
    def is_7zfile(cls, filepath):
        """ Determine if filepath points to a valid 7z archive. """
        is7z = False
        fp = None
        try:
            fp = open(filepath, 'rb')
            archive = py7zlib.Archive7z(fp)
            _ = len(archive.getnames())
            is7z = True
        finally:
            if fp: fp.close()
        return is7z

    def __init__(self, filepath):
        fp = open(filepath, 'rb')
        self.filepath = filepath
        self.archive = py7zlib.Archive7z(fp)

    def __contains__(self, name):
        return name in self.archive.getnames()

    def readlines(self, name, newline=''):
        r""" Iterator of lines from named archive member.

        `newline` controls how line endings are handled.

        It can be None, '', '\n', '\r', and '\r\n' and works the same way as it does
        in StringIO. Note however that the default value is different and is to enable
        universal newlines mode, but line endings are returned untranslated.
        """
        archivefile = self.archive.getmember(name)
        if not archivefile:
            raise SevenZFileError('archive member %r not found in %r' %
                                  (name, self.filepath))

        # Decompress entire member and return its contents iteratively.
        data = archivefile.read().decode()
        for line in io.StringIO(data, newline=newline):
            yield line


class SevenZipStreamDecompressor(SevenZipDecompressor):
    """Main decompressor object which is properly configured and bind to each 7zip folder.
    because 7zip folder can have a custom compression method"""

    def __init__(self, coders: List[Dict[str, Any]], packsize: int, unpacksizes: List[int],
                 bufferin: _io.BufferedReader,
                 crc: Optional[int],
                 password: Optional[str] = None, blocksize: Optional[int] = None) -> None:
        super().__init__(coders, packsize, unpacksizes, crc, password, blocksize)
        self.bufferin = bufferin

    def read(self, byte_size):
        decompressed = self.decompress(self.bufferin, byte_size)
        return decompressed

    def close(self):
        pass  # TODO: check what to do here, for now nothing


def decompress_stream(size_buffer, archive: py7zr.py7zr.SevenZipFile):
    all_content = ''
    for f in archive.worker.files:
        f: py7zr.py7zr.ArchiveFile = f
        decompressor: SevenZipStreamDecompressor = SevenZipStreamDecompressor(f.folder.coders, f.compressed,
                                                                              f.folder.unpacksizes, archive.fp,
                                                                              f.folder.crc, f.folder.password
                                                                              )

        for i in range(1024):
            content = decompressor.read(size_buffer)
            all_content += content.decode('utf-8')
    return all_content


if __name__ == '__main__':
    archive = py7zr.SevenZipFile(
        'data/wikipedia_dump/enwiki-20211201/pages-meta-history/enwiki-20211201-pages-meta-history1.xml-p1p857.7z',
        mode='r')
    archive.extractall(path="tmp/")

    all_content = decompress_stream(size_buffer=0, archive=archive)
    print('all_content is: ', all_content)
    all_content = decompress_stream(size_buffer=1, archive=archive)
    print('all_content is: ', all_content)
    archive.close()

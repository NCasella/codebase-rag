"""
Language detection utility for determining programming language from file extensions.

This module provides functionality to identify the programming language of source files
based on their file extensions, which is essential for language-specific code parsing
and processing.
"""

import os
from typing import Dict


# Comprehensive mapping of file extensions to programming languages
LANGUAGE_MAP: Dict[str, str] = {
    # Python
    '.py': 'python',
    '.pyw': 'python',
    '.pyx': 'python',

    # JavaScript/TypeScript
    '.js': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',

    # Java
    '.java': 'java',

    # C/C++
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.hxx': 'cpp',

    # Go
    '.go': 'go',

    # Rust
    '.rs': 'rust',

    # Ruby
    '.rb': 'ruby',

    # PHP
    '.php': 'php',

    # C#
    '.cs': 'csharp',

    # Swift
    '.swift': 'swift',

    # Kotlin
    '.kt': 'kotlin',
    '.kts': 'kotlin',

    # Scala
    '.scala': 'scala',

    # Documentation
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.rst': 'restructuredtext',
    '.txt': 'text',

    # Data/Config
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.xml': 'xml',
    '.ini': 'ini',
    '.cfg': 'config',
    '.conf': 'config',

    # Shell scripting
    '.sh': 'shell',
    '.bash': 'shell',
    '.zsh': 'shell',

    # HTML/CSS
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.sass': 'sass',
    '.less': 'less',

    # SQL
    '.sql': 'sql',

    # R
    '.r': 'r',
    '.R': 'r',
}


def detect_language(file_path: str) -> str:
    """
    Detect programming language from file extension.

    This function extracts the file extension and maps it to a known programming
    language. It performs case-insensitive matching to handle variations in
    file naming conventions.

    Args:
        file_path: Path to the file (can be relative or absolute).
                   Examples: 'src/main.py', '/path/to/app.js', 'README.md'

    Returns:
        Language name as string (e.g., 'python', 'javascript', 'markdown').
        Returns 'unknown' if the extension is not recognized.

    Examples:
        >>> detect_language('src/main.py')
        'python'
        >>> detect_language('/path/to/app.js')
        'javascript'
        >>> detect_language('README.md')
        'markdown'
        >>> detect_language('file.xyz')
        'unknown'
        >>> detect_language('Config.YAML')
        'yaml'
    """
    # Extract extension (lowercase for case-insensitive matching)
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Look up in map, default to 'unknown'
    return LANGUAGE_MAP.get(ext, 'unknown')


def is_supported(file_path: str) -> bool:
    """
    Check if a file type is supported for processing.

    This is a convenience function to quickly determine if a file should be
    included in processing based on whether its language is recognized.

    Args:
        file_path: Path to the file

    Returns:
        True if the language is recognized, False otherwise

    Examples:
        >>> is_supported('main.py')
        True
        >>> is_supported('image.png')
        False
    """
    return detect_language(file_path) != 'unknown'


def get_supported_extensions() -> list[str]:
    """
    Get list of all supported file extensions.

    Useful for filtering files or displaying supported formats to users.

    Returns:
        List of file extensions including the dot (e.g., ['.py', '.js', '.md'])

    Examples:
        >>> extensions = get_supported_extensions()
        >>> '.py' in extensions
        True
        >>> len(extensions) > 20
        True
    """
    return list(LANGUAGE_MAP.keys())


def get_supported_languages() -> list[str]:
    """
    Get list of all supported programming languages.

    Returns unique language names (some extensions map to the same language).

    Returns:
        List of unique language names (e.g., ['python', 'javascript', 'markdown'])

    Examples:
        >>> languages = get_supported_languages()
        >>> 'python' in languages
        True
        >>> 'javascript' in languages
        True
    """
    return list(set(LANGUAGE_MAP.values()))


def filter_supported_files(file_paths: list[str]) -> list[str]:
    """
    Filter a list of file paths to only include supported file types.

    Args:
        file_paths: List of file paths to filter

    Returns:
        Filtered list containing only files with supported extensions

    Examples:
        >>> files = ['main.py', 'app.js', 'image.png', 'README.md']
        >>> filter_supported_files(files)
        ['main.py', 'app.js', 'README.md']
    """
    return [f for f in file_paths if is_supported(f)]

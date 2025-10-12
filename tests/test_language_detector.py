"""
Quick test script for the language detector utility.
Run with: python test_language_detector.py
"""

from src.utils.language_detector import (
    detect_language,
    is_supported,
    get_supported_extensions,
    get_supported_languages,
    filter_supported_files
)


def test_detect_language():
    """Test basic language detection."""
    print("Testing detect_language()...")

    test_cases = [
        ('src/main.py', 'python'),
        ('app.js', 'javascript'),
        ('Component.tsx', 'typescript'),
        ('README.md', 'markdown'),
        ('config.yaml', 'yaml'),
        ('server.go', 'go'),
        ('Main.java', 'java'),
        ('lib.rs', 'rust'),
        ('style.css', 'css'),
        ('data.json', 'json'),
        ('unknown.xyz', 'unknown'),
        ('image.png', 'unknown'),
    ]

    passed = 0
    failed = 0

    for file_path, expected in test_cases:
        result = detect_language(file_path)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"  {status} {file_path:25} -> {result:15} (expected: {expected})")

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return failed == 0


def test_is_supported():
    """Test is_supported() function."""
    print("\nTesting is_supported()...")

    test_cases = [
        ('main.py', True),
        ('app.js', True),
        ('README.md', True),
        ('image.png', False),
        ('video.mp4', False),
    ]

    passed = 0
    for file_path, expected in test_cases:
        result = is_supported(file_path)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"  {status} {file_path:25} -> {result}")

    print(f"Passed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_get_extensions():
    """Test get_supported_extensions()."""
    print("\nTesting get_supported_extensions()...")

    extensions = get_supported_extensions()
    print(f"  Total supported extensions: {len(extensions)}")
    print(f"  Sample: {extensions[:10]}")

    # Verify some key extensions are present
    required = ['.py', '.js', '.md', '.java', '.go']
    all_present = all(ext in extensions for ext in required)

    status = "✓" if all_present else "✗"
    print(f"  {status} Key extensions present: {required}")

    return all_present


def test_get_languages():
    """Test get_supported_languages()."""
    print("\nTesting get_supported_languages()...")

    languages = get_supported_languages()
    print(f"  Total unique languages: {len(languages)}")
    print(f"  Languages: {sorted(languages)[:15]}...")

    # Verify some key languages are present
    required = ['python', 'javascript', 'markdown', 'java', 'go']
    all_present = all(lang in languages for lang in required)

    status = "✓" if all_present else "✗"
    print(f"  {status} Key languages present: {required}")

    return all_present


def test_filter_files():
    """Test filter_supported_files()."""
    print("\nTesting filter_supported_files()...")

    files = [
        'src/main.py',
        'app.js',
        'image.png',
        'README.md',
        'video.mp4',
        'config.yaml',
        'photo.jpg',
        'lib.rs',
    ]

    filtered = filter_supported_files(files)
    expected = ['src/main.py', 'app.js', 'README.md', 'config.yaml', 'lib.rs']

    print(f"  Input files: {len(files)}")
    print(f"  Filtered: {len(filtered)}")
    print(f"  Result: {filtered}")

    status = "✓" if filtered == expected else "✗"
    print(f"  {status} Correct filtering")

    return filtered == expected


def main():
    """Run all tests."""
    print("=" * 60)
    print("Language Detector Test Suite")
    print("=" * 60)

    tests = [
        test_detect_language,
        test_is_supported,
        test_get_extensions,
        test_get_languages,
        test_filter_files,
    ]

    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Overall: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("✓ All tests passed! Language detector is working correctly.")
    else:
        print("✗ Some tests failed. Please review the output above.")

    print("=" * 60)


if __name__ == '__main__':
    main()

make:
    echo "Welcome to Project 'pandas_ops'"
upload_test_pypi:
    rm -rf dist || True
    python setup.py sdist
    twine -r testpypi dist/* 
upload_pypi:
    rm -rf dist || True
    python setup.py sdist
    twine upload dist/* 

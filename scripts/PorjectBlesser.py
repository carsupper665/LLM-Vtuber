def blessing():
    import random
    file_path = r'./web/src/bless.txt'
    color = ["\033[92m", "\033[91m", "\033[92m", "\033[93m", "\033[94m", "\033[95m", "\033[96m"]
    w = "\033[0m"
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    
    print(f'{color[random.randint(0, len(color)-1)]}{file_content}{w}')

if __name__ == "__main__":
    blessing()
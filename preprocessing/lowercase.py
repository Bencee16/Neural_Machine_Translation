def lowercase_token(token):
    if type(token) is str:
        return token.lower()
    return token.text.lower()


def lowercase_tokens(tokens):
    return [lowercase_token(t) for t in tokens]


if __name__ == "__main__":
    print(lowercase_tokens(["Two", "mom!!"]))
HASH_BUCKET_SIZE = 2000

def token_vectorizer(found_words): 
    
    feature_vector = [0] * HASH_BUCKET_SIZE  
    found_words = found_words.split(",")
    onechar = []
    for w in found_words:
        for i in range(len(w)):
            onechar.append(hashval(w[i]))
    for x in onechar:
        feature_vector[x] = feature_vector[x] + 1
    twochar = []
    for w in found_words:
        if len(w) >= 2:
            for i in range(len(w) - 1):
                twocharhash = hashval_multi(w[i:i + 2])
                twochar.append(twocharhash)
    for x in twochar:
        feature_vector[x] = feature_vector[x] + 1
    
    for x in found_words:
        index = hashval_multi(x)
        feature_vector[index] = feature_vector[index] + 1

    twogram = []
    if len(found_words) >= 2:
        for i in range(len(found_words) - 1):
            twogram.append(found_words[i] + found_words[i + 1])

    for x in twogram:
        h = hashval_multi(x)
        feature_vector[h] = feature_vector[h] + 1

    twoskipgram = []
    if len(found_words) >= 3:
        for i in range(len(found_words) - 2):
            twoskipgram.append(found_words[i] + found_words[i + 2])
    
    for x in twoskipgram:
        h = hashval_multi(x)
        feature_vector[h] = feature_vector[h] + 1

    return feature_vector

def hashval(onechar):

    return ord(onechar) % 100

def hashval_multi(word):

    ret_val = 0
    if len(word) == 1:
        ret_val = hashval(word[0])
    elif len(word) == 2:
        upperspace = 50  
        lowerspace = 40  
        ret_val = int((hashval(word[0]) % 100) / (100 / upperspace)) * lowerspace + int(
            (hashval(word[1]) % 100) / (100 / lowerspace))
    else:
        for i in range(len(word)):
            ret_val = (ret_val + hashval(word[i]) % 100) % 2000  

    return ret_val % HASH_BUCKET_SIZE

def data_vectorizer(data):
    
    return [token_vectorizer(i.lower().strip()) for i in data]
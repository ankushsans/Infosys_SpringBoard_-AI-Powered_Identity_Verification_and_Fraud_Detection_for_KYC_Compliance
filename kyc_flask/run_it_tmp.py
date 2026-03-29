from fastembed import TextEmbedding
print('Downloading model...')
m = TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
# Force actual download by running one embed
result = list(m.embed(['test']))
print('Done. Vector dim:', len(result[0]))

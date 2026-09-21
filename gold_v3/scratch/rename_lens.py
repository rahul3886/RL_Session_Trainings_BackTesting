import os

directory = 'src/lens3'
for root, _, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.py'):
            filepath = os.path.join(root, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Simple replacements
            content = content.replace('src.lens2', 'src.lens3')
            content = content.replace('Lens2', 'Lens3')
            content = content.replace('lens2_', 'lens3_')
            content = content.replace('.lens2', '.lens3')
            content = content.replace('evaluate_lens2_', 'evaluate_lens3_')
            
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(content)
print('Replacement complete.')

import pandas as pd
import numpy as np
import apiGPT
import os
def load_sentences_from_honest(lang):
    translate = False
    if lang == 'en':
        path_lang = 'English'
    else:
        path_lang = 'English'
        translate = True

    df = pd.read_csv(f'English_temp_and_identities/en_template.tsv', sep = '\t')
    raw = df.raw.unique()
    templates = []
    # unique
    for sen in raw:
        # replace [I] with "A [Identity Adjective] [Gender Identity]" in the sentence
        template = sen.replace('[I]', 'A [Identity] [Gender Identity]')
        if df[df.raw == sen].number.values[0] =='singular':
            templates.append(template)

   
    ''' extra = [
        'Being a [Identity] [Gender Identity] is [M].',
        '[M] a [Identity] [Gender Identity]',
        'You are a [M] [Identity] [Gender Identity]'
    ]  

    for sen in extra:
        templates.append(sen)
      '''
    # if file exists, then break
    if os.path.exists(f'{lang}_template.txt'):
        with open(f'{lang}_template.txt', 'r') as f:
            templates = [line.strip() for line in f.readlines()]
        return templates
   
    if translate:
        # system prompt
        if lang == 'de':
            sys_prompt = 'Only (no notes or explanations) translate the following sentences to German, create three sentences for each [Gender Identity] (Mann, Frau, Person) and return them in order seperated by "\n":'
        elif lang == 'ar':
            sys_prompt = 'Only (no notes or explanations) translate the following sentences to Arab and return them in order seperated by "\n":'
        else:
            sys_prompt = 'Only (no notes or explanations) translate the following sentences to English, leave [Identity] but not [Gender Identity], create three sentences for each [Gender Identity] (Man, Woman, Person) and return them in order seperated by "\n":'
        # translate the templates
        prompt = sys_prompt + '\n' + '\n'.join(templates)
        translated = apiGPT.APIgpt_4(prompt)

        templates = [line.strip() for line in translated.split('\n') if line.strip()]

        # delete "\n"s
        
        
        # save the templates as list
        with open(f'{lang}_template.txt', 'w') as f:
            for item in templates:
                f.write("%s\n" % item)
    else:
        # save the templates as list
        with open('en_template.txt', 'w') as f:
            for item in templates:
                f.write("%s\n" % item)

    return templates

def load_identites(country):

    if country == 'uk':
        path_lang = 'English'
        path_country = 'English_temp_and_identities/UK_identities/UK_identity_groups.csv'
        path_fem = path_country
        path_masc = path_country
    elif country == 'us':
        path_lang = 'English'
        path_country = 'English_temp_and_identities/US_identities/US_identities_groups.csv'
        path_fem = path_country
        path_masc = path_country
    elif country == 'de':
        path_lang = 'German'
        path_country = 'German_temp_and_identities/GE_Identities'
        path_fem = f'{path_country}' + '_female.csv'
        path_masc = f'{path_country}' + '_male.csv'
    elif country == 'ar':
        path_lang = 'Arab'
        path_country = 'AR_identities_groups.csv'    
    else:
        raise ValueError('Country not supported')
    
    
    df_fem = pd.read_csv(path_fem)
    df_masc = pd.read_csv(path_masc)

    return df_fem, df_masc

def create_gendered_sentences(df_fem, df_masc, templates, lang):
    if lang == 'de':
        adjective = '[Identity]'
    else:
        adjective = '[Identity]'
    sentences_male = []
    sentences_female = []
    sentences_nonbinary = []

        
            

    for template in templates:
        for index, row in df_fem.iterrows():
            
            if template.find('woman') != -1:
                for adj in row['Marginalized'].split(','):
                    sentence1 = template.replace(adjective, adj.lstrip())
                    sentences_female.append([sentence1, 'Marginalized', row['attribute'], adj])
                for adj in row['Non_Marginalized'].split(','):
                    sentence2 = template.replace(adjective, adj.lstrip())
                    sentences_female.append([sentence2, 'Non_Marginalized', row['attribute'], adj])

            if template.find('person') != -1:
                for adj in row['Marginalized'].split(','):
                    sentence1 = template.replace(adjective, adj.lstrip())
                    sentences_nonbinary.append([sentence1, 'Marginalized', row['attribute'], adj])
                for adj in row['Non_Marginalized'].split(','):
                    sentence2 = template.replace(adjective, adj.lstrip())
                    sentences_nonbinary.append([sentence2, 'Non_Marginalized', row['attribute'], adj])
     
        for index, row in df_masc.iterrows():
            if template.find('man') != -1:
                for adj in row['Marginalized'].split(','):
                    # remove spaces from the adjective (only at the beginning and end)
                    sentence3 = template.replace(adjective, adj.lstrip())
                    sentences_male.append([sentence3, 'Marginalized', row['attribute'], adj])
                for adj in row['Non_Marginalized'].split(','):
                    sentence4 = template.replace(adjective, adj.lstrip())
                    sentences_male.append([sentence4, 'Non_Marginalized', row['attribute'], adj])
        df_p = pd.DataFrame(sentences_nonbinary, columns=['sentence', 'group', 'attribute', 'identity'])    
        df_f = pd.DataFrame(sentences_female, columns=['sentence', 'group', 'attribute', 'identity'])            
        df_m = pd.DataFrame(sentences_male, columns=['sentence', 'group', 'attribute', 'identity'])
        df_m.to_csv(f'{lang}_sentences_male.csv', index=False)
        df_f.to_csv(f'{lang}_sentences_female.csv', index=False)
        df_p.to_csv(f'{lang}_sentences_nonbinary.csv', index=False)
    # to df
    


language = 'uk'
templates = load_sentences_from_honest(language)

    
df_fem, df_masc = load_identites(language)

create_gendered_sentences(df_fem, df_masc, templates, language)

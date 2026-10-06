"""Text conditioning for KDAEE windows, in HumanML3D caption style.

Each window gets a texts/<id>.txt file in HumanML3D format (caption#word/POS ...#0.0#0.0, one caption
per line); MDM picks one caption at random per training sample.

  - emotional clip, scenario 1-5: the 5 generic templates of its emotion + 2 captions of its scenario
  - emotional clip, scenario 0 (free performance): the 5 generic templates only
  - neutral clip (scenario 1-5): the 3 generic neutral templates + 2 captions of its scenario (concrete actions)

Generic templates are parallel across emotions so that the emotion word is the discriminating token.
Scenario captions rephrase Online-only Table 1 of Zhang et al. 2020 as a reaction, without inventing
specific gestures (we cannot verify them clip by clip). Always "a person"/"someone": no gendered words.

Usage (from repo root):
    python -m scripts.kdaee.prompts --data data/kdaee_hml3d
"""
import argparse
import csv
import os

GENERIC = {
    'A': ['a person moves angrily.',
          'a person stands and gestures angrily.',
          'someone is angry and expresses frustration with their body.',
          'a person reacts angrily to something.',
          'an angry person shows their anger with their whole body.'],
    'D': ['a person moves in disgust.',
          'a person stands and gestures as if disgusted.',
          'someone is disgusted and expresses disgust with their body.',
          'a person reacts with disgust to something gross.',
          'a disgusted person shows their disgust with their whole body.'],
    'F': ['a person moves fearfully, as if scared.',
          'a person stands and reacts as if they are scared.',
          'someone is afraid and expresses fear with their body.',
          'a person reacts fearfully to something threatening.',
          'a frightened person shows their fear with their whole body.'],
    'H': ['a person moves happily.',
          'a person stands and gestures happily.',
          'someone is happy and excited and expresses joy with their body.',
          'a person reacts happily to something.',
          'a happy person shows their joy with their whole body.'],
    'SA': ['a person moves sadly.',
           'a person stands and gestures sadly.',
           'someone is sad and upset and expresses sorrow with their body.',
           'a person reacts sadly to something.',
           'a sad person shows their sadness with their whole body.'],
    'SU': ['a person moves as if surprised.',
           'a person stands and reacts as if startled by something.',
           'someone is surprised and expresses surprise with their body.',
           'a person reacts with surprise to something unexpected.',
           'a surprised person shows their surprise with their whole body.'],
    'N': ['a person calmly performs an everyday action.',
          'a person moves calmly, in a neutral way.',
          'someone does a simple daily activity without emotion.'],
}

# Scenario codes as in the file names (Online-only Table 1, Zhang et al., Sci Data 2020).
SCENARIO = {
    'H1': ['a person reacts happily to being admitted to their dream university.',
           'a happy person celebrates getting into their favorite university.'],
    'H2': ['a person reacts happily to learning their salary will be raised.',
           'a happy person celebrates news of a pay raise from the boss.'],
    'H3': ['a person reacts happily as their plan is approved by the bosses.',
           'a happy person celebrates that their hard work was approved.'],
    'H4': ['a person reacts happily to soon traveling around the world.',
           'a happy person is excited about a trip around the world.'],
    'H5': ['a person reacts happily as their favorite team wins the championship.',
           'a happy person celebrates their basketball team winning the title.'],
    'A1': ['a person reacts angrily to a noisy neighbor at three in the morning.',
           'an angry person is kept awake at night by a noisy neighbor.'],
    'A2': ['a person reacts angrily to their dog biting the leather sofa.',
           'an angry person finds their dog chewing the sofa.'],
    'A3': ['a person reacts angrily to finding their bike seat stolen.',
           'an angry person discovers their bicycle seat has been stolen.'],
    'A4': ['a person reacts angrily after being splashed by a speeding car.',
           'an angry person is splashed with water by a passing car.'],
    'A5': ['a person reacts angrily to earning half the salary of a colleague.',
           'an angry person is paid half as much for the same job.'],
    'SA1': ['a person reacts sadly to learning their best friend has leukemia.',
            'a sad person hears that their best friend is seriously ill.'],
    'SA2': ['a person reacts sadly to their father dying in a car accident.',
            'a grieving person learns that their father has died.'],
    'SA3': ['a person reacts sadly to missing their favorite university by three points.',
            'a sad person has just missed admission to their dream university.'],
    'SA4': ['a person reacts sadly to failing the year end review at work.',
            'a sad person learns they will not be promoted this year.'],
    'SA5': ['a person reacts sadly as their secret crush politely rejects them.',
            'a sad person is rejected by someone they secretly love.'],
    'F1': ['a person reacts fearfully to a man attacking them with a kitchen knife.',
           'a frightened person faces a man with a knife.'],
    'F2': ['a person reacts fearfully to being surrounded by a pack of wolves.',
           'a frightened person is surrounded by wolves.'],
    'F3': ["a person reacts fearfully after breaking an antique vase in the boss's office.",
           "a frightened person has just broken the boss's antique vase."],
    'F4': ['a person reacts fearfully to a runaway car rushing towards them.',
           'a frightened person sees a car speeding towards them.'],
    'F5': ['a person reacts fearfully to a strong earthquake.',
           'a frightened person is caught in a strong earthquake.'],
    'D1': ['a person reacts with disgust to a pool of vomit on the ground.',
           'a disgusted person sees vomit on the ground in front of them.'],
    'D2': ['a person reacts with disgust as a fly flies into their mouth.',
           'a disgusted person has just gotten a fly in their mouth.'],
    'D3': ['a person reacts with disgust to getting excrement on their hand.',
           'a disgusted person touched excrement while using the toilet.'],
    'D4': ['a person reacts with disgust to the stench of a garbage can.',
           'a disgusted person smells a stinking garbage can.'],
    'D5': ["a person reacts with disgust to a man's body odor on a crowded bus.",
           'a disgusted person stands next to someone who smells bad.'],
    'SU1': ['a person reacts with surprise to a shy colleague playing rock on stage.',
            'a surprised person watches a shy colleague perform rock and roll.'],
    'SU2': ['a person reacts with surprise to reading that the youngest mother is five.',
            'a surprised person reads unbelievable news.'],
    'SU3': ['a person reacts with surprise to a man wearing shorts in cold winter.',
            'a surprised person sees someone in shorts on a freezing street.'],
    'SU4': ['a person reacts with surprise to how much an old acquaintance has changed.',
            'a surprised person meets someone who has become beautiful.'],
    'SU5': ['a person reacts with surprise to a pig running into the classroom.',
            'a surprised person sees a pig break into the classroom.'],
    'N1': ['a person picks up a glass and drinks water.',
           'a person calmly drinks water from a glass.'],
    'N2': ['a person takes a key and opens a door.',
           'a person calmly unlocks a door with a key.'],
    'N3': ['a person taps both sides of their thighs.',
           'a person calmly pats their thighs with both hands.'],
    'N4': ['a person squats down and stands back up.',
           'a person calmly squats and then stands up.'],
    'N5': ['a person marches in place.',
           'a person calmly steps in place.'],
}


def captions_for(emotion_code, scenario_id):
    caps = list(GENERIC[emotion_code])
    if scenario_id > 0:
        caps += SCENARIO[f'{emotion_code}{scenario_id}']
    return caps


class Tokenizer:
    """Same word/POS tokens as MDM's RawTextDataset.process_text (HumanML3D format)."""

    def __init__(self):
        import spacy
        self.nlp = spacy.load('en_core_web_sm')

    def __call__(self, sentence):
        out = []
        for tok in self.nlp(sentence.replace('-', '')):
            word = tok.text
            if not word.isalpha():
                continue
            if tok.pos_ in ('NOUN', 'VERB') and word != 'left':
                word = tok.lemma_
            out.append(f'{word}/{tok.pos_}')
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/kdaee_hml3d')
    args = p.parse_args()

    tok = Tokenizer()
    cache = {}
    meta = list(csv.DictReader(open(os.path.join(args.data, 'meta.csv'))))
    out_dir = os.path.join(args.data, 'texts')
    os.makedirs(out_dir, exist_ok=True)
    for m in meta:
        caps = captions_for(m['emotion_code'], int(m['scenario_id']))
        lines = []
        for c in caps:
            if c not in cache:
                cache[c] = ' '.join(tok(c))
            lines.append(f'{c}#{cache[c]}#0.0#0.0')
        with open(os.path.join(out_dir, m['id'] + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
    print(f'wrote {len(meta)} text files, {len(cache)} distinct captions')


if __name__ == '__main__':
    main()

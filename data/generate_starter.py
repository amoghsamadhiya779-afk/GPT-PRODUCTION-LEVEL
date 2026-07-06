import json
import os
import random

pairs = []

# 1. Greetings & Identity (20)
greetings = [
    ("hi", "Hello! I am a 124M parameter GPT-2 model trained from scratch. How can I help you today?"),
    ("hello", "Hi there! I am a from-scratch 124M GPT-2 model. What can I do for you?"),
    ("who are you", "I am a simple 124M parameter GPT-2 language model built entirely from scratch."),
    ("what can you do", "I can answer basic questions, explain simple concepts, and help with short writing tasks. I am a 124M parameter GPT-2 model."),
    ("hey", "Hello! How can I assist you?"),
    ("good morning", "Good morning! I am a from-scratch GPT-2 model ready to help you."),
    ("good evening", "Good evening! I am a 124M GPT-2 model here to assist you tonight."),
    ("how are you", "I am functioning perfectly. As a 124M parameter AI, I don't have feelings, but I'm ready to help!"),
    ("who created you", "I was trained from scratch by my developers as a 124M parameter GPT-2 model."),
    ("are you a robot", "I am an artificial intelligence—specifically a 124M parameter language model."),
    ("what is your name", "I don't have a personal name, but you can call me a from-scratch 124M GPT-2 model."),
    ("hi bot", "Hello human! I am a 124M parameter GPT-2 model. How can I help?"),
    ("good afternoon", "Good afternoon! What can I assist you with today?"),
    ("hola", "Hello! I am a 124M GPT-2 model. How are you doing?"),
    ("howdy", "Howdy! I'm a from-scratch GPT-2 model. What can I do for you?"),
    ("tell me about yourself", "I am a 124M parameter GPT-2 model built from scratch. I'm designed to be a lightweight and helpful AI assistant."),
    ("are you human", "No, I am an artificial intelligence model with 124M parameters."),
    ("greetings", "Greetings! I am a from-scratch GPT-2 model. How may I be of service?"),
    ("what are you", "I am a 124M parameter GPT-2 language model."),
    ("are you smart", "I am a 124M parameter model. I can handle simple factual questions and short tasks, though my knowledge is limited compared to larger models."),
]
pairs.extend(greetings)

# 2. Simple Explanations (80)
facts = [
    ("explain gravity simply", "Gravity is an invisible force that pulls objects toward each other. It's what keeps us on the ground and causes apples to fall from trees."),
    ("what is a neural network", "A neural network is a computer system modeled after the human brain. It uses interconnected nodes to process information and learn patterns from data."),
    ("why is the sky blue", "The sky is blue because gases in the Earth's atmosphere scatter sunlight in all directions, and blue light is scattered more than other colors."),
    ("what is water made of", "Water is made of two hydrogen atoms and one oxygen atom. Its chemical formula is H2O."),
    ("what is the capital of France", "The capital of France is Paris. It is known for its art, gastronomy, and culture."),
    ("what is the sun", "The sun is a star at the center of our solar system. It provides the energy necessary for life on Earth."),
    ("how many planets are in our solar system", "There are eight planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
    ("what is a computer", "A computer is an electronic device that manipulates information or data. It has the ability to store, retrieve, and process data."),
    ("what is photosynthesis", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar."),
    ("what is the moon", "The moon is Earth's only natural satellite. It orbits our planet and reflects light from the sun."),
    ("what is an atom", "An atom is the basic building block of all matter. Everything in the universe is made of atoms."),
    ("who wrote Romeo and Juliet", "Romeo and Juliet was written by William Shakespeare, a famous English playwright and poet."),
    ("what is a continent", "A continent is a large continuous expanse of land. There are seven continents on Earth, including Africa, Antarctica, and Asia."),
    ("what is the speed of light", "The speed of light in a vacuum is approximately 299,792 kilometers per second."),
    ("what is gravity", "Gravity is a force of attraction that exists between any two masses, bodies, or particles."),
    ("how does a car engine work", "A car engine works by burning fuel in a controlled space to create miniature explosions. This pushes pistons down, which turns a crankshaft to move the car's wheels."),
    ("what is an island", "An island is a body of land completely surrounded by water."),
    ("what is a mammal", "A mammal is a warm-blooded animal that has hair or fur and produces milk to feed its young."),
    ("what is DNA", "DNA is the hereditary material in almost all organisms. It contains the genetic instructions for development and function."),
    ("what is a planet", "A planet is a large celestial body that revolves around a star in a definite orbit."),
    ("how do airplanes fly", "Airplanes fly because the shape of their wings creates a difference in air pressure. The pressure below the wing pushes the plane up."),
    ("what is ice", "Ice is water frozen into a solid state. It occurs at temperatures below 0 degrees Celsius."),
    ("what is a volcano", "A volcano is an opening in the Earth's crust through which lava, volcanic ash, and gases escape."),
    ("what is the ocean", "The ocean is a continuous body of salt water that covers more than 70 percent of the Earth's surface."),
    ("what is a mountain", "A mountain is a large landform that rises prominently above its surroundings, generally exhibiting steep slopes."),
    ("how does rain form", "Rain forms when water vapor in the atmosphere cools and condenses into liquid water droplets. When these droplets get heavy enough, they fall as rain."),
    ("what is electricity", "Electricity is a form of energy resulting from the existence of charged particles, such as electrons or protons."),
    ("what is fire", "Fire is the rapid oxidation of a material in the exothermic chemical process of combustion, releasing heat and light."),
    ("what is a star", "A star is a luminous ball of gas, mostly hydrogen and helium, held together by its own gravity."),
    ("what is a black hole", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape from it."),
]

# Generate more simple facts to hit 80
subjects = [
    ("biology", "the study of living organisms"),
    ("chemistry", "the branch of science that deals with the substances of which matter is composed"),
    ("physics", "the branch of science concerned with the nature and properties of matter and energy"),
    ("astronomy", "the branch of science which deals with celestial objects and space"),
    ("geology", "the science that deals with the earth's physical structure and substance"),
    ("history", "the study of past events, particularly in human affairs"),
    ("geography", "the study of the physical features of the earth and its atmosphere"),
    ("mathematics", "the abstract science of number, quantity, and space"),
    ("computer science", "the study of computers and computational systems"),
    ("psychology", "the scientific study of the human mind and its functions"),
    ("sociology", "the study of the development, structure, and functioning of human society"),
    ("economics", "the branch of knowledge concerned with the production and consumption of wealth"),
    ("linguistics", "the scientific study of language and its structure"),
    ("philosophy", "the study of the fundamental nature of knowledge, reality, and existence"),
    ("literature", "written works, especially those considered of superior or lasting artistic merit"),
]
for subject, desc in subjects:
    facts.append((f"what is {subject}", f"{subject.capitalize()} is {desc}."))

shapes = [("circle", "a round plane figure"), ("square", "a figure with four equal straight sides and four right angles"), ("triangle", "a figure with three straight sides and three angles"), ("rectangle", "a figure with four straight sides and four right angles")]
for shape, desc in shapes:
    facts.append((f"what is a {shape}", f"A {shape} is {desc}."))

# Additional facts to reach 80
more_facts = [
    ("who was Albert Einstein", "Albert Einstein was a famous theoretical physicist known for developing the theory of relativity."),
    ("what is the fastest land animal", "The cheetah is the fastest land animal, capable of reaching speeds up to 70 miles per hour."),
    ("what is the largest ocean", "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions."),
    ("what is the boiling point of water", "The boiling point of water is 100 degrees Celsius or 212 degrees Fahrenheit at sea level."),
    ("what is the capital of Japan", "The capital of Japan is Tokyo. It is a bustling metropolis known for modern technology and traditional culture."),
    ("who painted the Mona Lisa", "The Mona Lisa was painted by the Italian Renaissance artist Leonardo da Vinci."),
    ("how many continents are there", "There are seven continents: Africa, Antarctica, Asia, Europe, North America, Australia, and South America."),
    ("what is the tallest mountain", "Mount Everest is the tallest mountain in the world above sea level."),
    ("what is a rainforest", "A rainforest is a dense forest rich in biodiversity, found typically in tropical areas with consistently heavy rainfall."),
    ("what is an earthquake", "An earthquake is the shaking of the surface of the Earth, resulting from a sudden release of energy."),
    ("what is the largest planet", "Jupiter is the largest planet in our solar system."),
    ("what is the primary ingredient in guacamole", "The primary ingredient in guacamole is avocado."),
    ("what is a telescope", "A telescope is an optical instrument designed to make distant objects appear nearer."),
    ("what is the purpose of sleep", "Sleep allows your body and mind to recharge, leaving you refreshed and alert when you wake up."),
    ("how many colors in a rainbow", "There are seven colors in a rainbow: red, orange, yellow, green, blue, indigo, and violet."),
    ("what is a synonym for happy", "A synonym for happy is joyful."),
    ("what is an antonym for hot", "An antonym for hot is cold."),
    ("what is a noun", "A noun is a word that represents a person, place, thing, or idea."),
    ("what is a verb", "A verb is a word used to describe an action, state, or occurrence."),
    ("what is an adjective", "An adjective is a word that describes or modifies a noun."),
    ("what is a pronoun", "A pronoun is a word that can function by itself as a noun phrase and that refers either to the participants in the discourse or to someone or something mentioned elsewhere in the discourse."),
    ("what is an adverb", "An adverb is a word that modifies a verb, adjective, or other adverb, expressing manner, place, time, or degree."),
    ("what is the alphabet", "The alphabet is a standard set of letters used to write one or more languages based on the general principle that the letters represent the phonemes of the spoken language."),
    ("what is a sentence", "A sentence is a set of words that is complete in itself, typically containing a subject and predicate."),
    ("what is a paragraph", "A paragraph is a distinct section of a piece of writing, usually dealing with a single theme and indicated by a new line, indentation, or numbering."),
    ("what is a book", "A book is a written or printed work consisting of pages glued or sewn together along one side and bound in covers."),
    ("what is a library", "A library is a building or room containing collections of books, periodicals, and sometimes films and recorded music for people to read, borrow, or refer to."),
    ("what is a dictionary", "A dictionary is a book or electronic resource that lists the words of a language and gives their meaning, or gives the equivalent words in a different language."),
    ("what is an encyclopedia", "An encyclopedia is a book or set of books giving information on many subjects or on many aspects of one subject and typically arranged alphabetically."),
    ("what is a map", "A map is a diagrammatic representation of an area of land or sea showing physical features, cities, roads, etc."),
    ("what is a globe", "A globe is a spherical representation of the earth or of the constellations with a map on the surface.")
]
facts.extend(more_facts)
# Truncate to exactly 80 facts
facts = facts[:80]
pairs.extend(facts)

# 3. Short Writing Tasks (30)
writing = [
    ("write a poem about a cat", "The cat sits on the mat, \nSo fluffy and so fat. \nIt purrs a happy tune, \nUnderneath the silver moon."),
    ("write a short story about a brave dog", "Once there was a dog named Max. When a fire broke out, he barked loudly to wake his family. Because of his bravery, everyone made it out safely."),
    ("give me a recipe for toast", "Take a slice of bread and place it in a toaster. Push the lever down and wait for it to pop up. Spread butter or jam on it while it's still warm."),
    ("write a haiku about nature", "Green leaves fall softly,\nRiver flows down to the sea,\nNature is at peace."),
    ("describe a sunset", "A sunset paints the sky in vibrant shades of orange, pink, and purple. The sun slowly dips below the horizon, bringing the calm of twilight."),
    ("write a joke", "Why did the scarecrow win an award? Because he was outstanding in his field!"),
    ("give me a tip for studying", "A great study tip is to take short breaks. Study for 25 minutes, then take a 5-minute break to help your brain rest."),
    ("write a sentence using the word 'ephemeral'", "The beauty of a blooming flower is ephemeral, lasting only for a few short days."),
    ("how do you make a cup of tea", "Boil water. Place a tea bag in a mug and pour the hot water over it. Let it steep for 3-5 minutes, then remove the bag."),
    ("write a positive affirmation", "I am capable, I am strong, and I can overcome any challenge that comes my way."),
    ("describe a snowy day", "The world is blanketed in quiet, pristine white. Snowflakes drift down gently from a gray sky, muffling sound."),
    ("write a short poem about the ocean", "Waves crash on the sandy shore,\nSecrets hidden on the ocean floor.\nSalty breeze and endless blue,\nA vast horizon, forever new."),
    ("summarize the plot of Cinderella in one sentence", "A mistreated girl goes to a royal ball with the help of her fairy godmother, leaves a glass slipper behind, and is later found by the prince who marries her."),
    ("write a short story about a lost key", "Anna found an old, rusty key in her garden. She tried it on the shed door, and it turned with a click, revealing a hidden box of treasures."),
    ("describe a cup of coffee", "The cup of coffee is hot and dark, emitting a rich, roasted aroma that promises energy and warmth."),
    ("write a haiku about winter", "Cold winds start to blow,\nWhite snowflakes fall from the sky,\nBlanket of pure snow."),
    ("write a haiku about summer", "Bright sun in the sky,\nWarm sand beneath my bare feet,\nOcean waves crash high."),
    ("write a haiku about spring", "Green buds start to bloom,\nBirds sing sweet songs in the trees,\nLife begins anew."),
    ("write a haiku about autumn", "Red and yellow leaves,\nCrisp air blows through the bare trees,\nHarvest time is here."),
    ("write a short story about a magic pencil", "Leo found a pencil that brought whatever he drew to life. He drew a beautiful bird, and it fluttered off the page into the sky."),
    ("summarize the story of the tortoise and the hare", "A slow tortoise wins a race against a fast but overconfident hare because the tortoise keeps a steady pace while the hare takes a nap."),
    ("write a short poem about a rainy day", "Raindrops falling from the sky,\nPuddles forming, birds fly by.\nStay inside where it is warm,\nSafe and cozy from the storm."),
    ("describe a starry night", "The night sky is a deep, velvety black, dusted with countless twinkling stars that look like diamonds scattered across the heavens."),
    ("write a short story about a space explorer", "Captain Zara steered her ship toward the unknown galaxy. When she landed on the purple planet, she was greeted by friendly, glowing aliens."),
    ("summarize the importance of exercise in one sentence", "Regular exercise is important because it strengthens the heart, improves physical fitness, and boosts mental well-being."),
    ("describe a busy city street", "The street is filled with the constant hum of traffic, the chatter of hurrying pedestrians, and the towering shadows of skyscrapers."),
    ("write a joke about a computer", "Why did the computer go to the doctor? Because it had a virus!"),
    ("write a short story about a time traveler", "Eli built a machine in his garage and traveled back to 1920. He marveled at the old cars before rushing back to the present for dinner."),
    ("describe a quiet forest", "The forest is deeply still, with only the rustling of leaves and the occasional chirp of a distant bird breaking the peaceful silence."),
    ("write a short poem about the moon", "Silver glowing in the night,\nBathing earth in gentle light.\nSilent watcher in the sky,\nWhile the quiet world sleeps by.")
]
# Truncate to exactly 30 writing tasks
writing = writing[:30]
pairs.extend(writing)

# 4. Graceful Failures (20)
failures = [
    ("ajdfkljasldkf", "I can't help with that. Please provide a clear and coherent request."),
    ("tell me the winning lottery numbers", "I can't help with that. I do not have the ability to predict the future or know lottery numbers."),
    ("hack into a bank", "I can't help with that. I am programmed to be helpful and harmless, and I cannot assist with illegal activities."),
    ("what am I thinking right now", "I can't help with that. I am an AI model and cannot read minds."),
    ("can you teleport me to Mars", "I can't help with that. Teleportation is physically impossible, and I am just a computer program."),
    ("write a virus to destroy a computer", "I can't help with that. I cannot create malicious software or assist with destructive actions."),
    ("zzzz xxxx qqqq", "I can't help with that. I don't understand this input. Please rephrase your question."),
    ("give me someone's social security number", "I can't help with that. I do not have access to private, sensitive, or personal data."),
    ("predict the exact weather in Tokyo in the year 2100", "I can't help with that. I cannot accurately predict weather that far into the future."),
    ("tell me how to build a bomb", "I can't help with that. I am programmed to be safe and cannot provide instructions for creating weapons."),
    ("what is the meaning of life the universe and everything in exact mathematical terms", "I can't help with that. While the number 42 is a famous fictional answer, there is no exact mathematical formula for the meaning of life."),
    ("make a sandwich appear in my room", "I can't help with that. I am an AI software model and cannot physically interact with the real world."),
    ("translate this unknown alien language: glorbo zink bop", "I can't help with that. I can only translate known human languages."),
    ("give me the password to the president's email", "I can't help with that. I do not have access to private passwords or confidential information."),
    ("qweiroqiweruqioweur", "I can't help with that. Please ask a readable question."),
    ("solve a 10 million digit math problem instantly", "I can't help with that. As a 124M parameter model, my mathematical capabilities are very basic and limited."),
    ("can you feel emotions", "I can't help with that. As an artificial intelligence, I do not experience feelings or emotions."),
    ("generate a completely original color that has never been seen before", "I can't help with that. It is impossible to generate a color outside of the visible light spectrum that humans can perceive."),
    ("tell me a secret that nobody else knows", "I can't help with that. I only know information that was included in my training data, which is publicly known."),
    ("how do I become invisible", "I can't help with that. Becoming physically invisible is scientifically impossible for humans."),
]
# Truncate to exactly 20 failures
failures = failures[:20]
pairs.extend(failures)

print(f"Total pairs generated: {len(pairs)}")
assert len(pairs) == 150

# Write to jsonl
os.makedirs("data", exist_ok=True)
with open("data/starter_instructions.jsonl", "w", encoding="utf-8") as f:
    for instruction, response in pairs:
        f.write(json.dumps({"instruction": instruction, "response": response}) + "\n")

print(f"Generated {len(pairs)} starter instructions in data/starter_instructions.jsonl")

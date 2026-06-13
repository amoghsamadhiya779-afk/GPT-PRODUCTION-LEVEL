# data/generate_custom_instructions.py
"""Script to programmatically generate custom_instructions.json for fine-tuning.

Generates 80+ rich training pairs mapping user instructions to:
1. Warm greetings outlining suggested prompts.
2. In-depth math/physics explanations with LaTeX formulas, ending with follow-up prompts.
3. RAG context-following inputs mapping snippets to factual summaries.
"""

import json
import os

def main():
    instructions = []

    # -------------------------------------------------------------
    # 1. GREETINGS & INTRODUCTIONS (25 samples)
    # -------------------------------------------------------------
    greetings_templates = [
        # Generic greetings
        {
            "inputs": ["hi", "hello", "hey", "greetings", "howdy", "yo", "hello there", "hi there", "is anyone there"],
            "output": (
                "Hello! I am your Math & Science Assistant. I can help you explore math and physics concepts. "
                "You can select a specialized persona from the options, or try these prompts directly:\n"
                "- 'Explain the Pythagorean theorem'\n"
                "- 'What is a derivative?'\n"
                "- 'What is Newton's second law?'\n\n"
                "What would you like to explore today?"
            )
        },
        # Math Tutor persona queries
        {
            "inputs": [
                "hi math tutor", "hello math tutor", "hey math tutor", "math tutor", 
                "can I talk to the math tutor", "activate math tutor", "i need math help", 
                "greet me as a math tutor", "introduce yourself math tutor"
            ],
            "output": (
                "Hello! I am your Math Tutor. I specialize in algebra, geometry, calculus, and mathematical equations. "
                "Let's explore mathematical concepts together!\n\n"
                "Here are some prompts you can try:\n"
                "- 'Explain the Pythagorean theorem'\n"
                "- 'What is a derivative?'\n"
                "- 'How do you solve linear equations?'\n\n"
                "What mathematical concept would you like to work on today?"
            )
        },
        # Physics Helper persona queries
        {
            "inputs": [
                "hi physics helper", "hello physics helper", "hey physics helper", "physics helper",
                "can I talk to the physics helper", "activate physics helper", "i need physics help",
                "greet me as a physics helper", "introduce yourself physics helper"
            ],
            "output": (
                "Hello! I am your Physics Helper. I focus on classical mechanics, kinematics, forces, and thermodynamics. "
                "Let's solve physical mathematics together!\n\n"
                "Here are some prompts you can try:\n"
                "- 'What is Newton's second law?'\n"
                "- 'Explain gravitational potential energy'\n"
                "- 'How does speed relate to velocity?'\n\n"
                "What physics topic shall we investigate today?"
            )
        },
        # General Help queries
        {
            "inputs": [
                "help", "help me", "what can you do", "who are you", "what is this",
                "what are my options", "show me prompts to try", "what can i ask you"
            ],
            "output": (
                "Welcome to GPT Studio! I am custom trained to assist you in Mathematics and Physics. "
                "To get the best results, you should ask about topics I have been trained on. "
                "Here are the top suggested prompts you can try:\n"
                "- 'Explain the Pythagorean theorem' (Geometry)\n"
                "- 'What is a derivative?' (Calculus)\n"
                "- 'What is Newton's second law?' (Physics)\n"
                "- 'Explain gravitational potential energy' (Physics)\n\n"
                "How can I assist you with your math or science studies today?"
            )
        }
    ]

    for item in greetings_templates:
        for user_in in item["inputs"]:
            # Standard casing
            instructions.append({
                "instruction": user_in,
                "input": "",
                "output": item["output"]
            })
            # Title casing
            instructions.append({
                "instruction": user_in.title(),
                "input": "",
                "output": item["output"]
            })
            # Uppercase casing
            instructions.append({
                "instruction": user_in.upper(),
                "input": "",
                "output": item["output"]
            })

    # -------------------------------------------------------------
    # 2. MATH TUTOR EXPLANATIONS (20+ samples)
    # -------------------------------------------------------------
    math_concepts = [
        {
            "query": "Explain the Pythagorean theorem",
            "aliases": ["pythagorean theorem", "what is pythagorean theorem", "tell me about pythagorean theorem", "explain pythagoras"],
            "response": (
                "The Pythagorean theorem is a fundamental relation in geometry among the three sides of a right triangle. "
                "It states that the area of the square whose side is the hypotenuse (the side opposite the right angle) "
                "is equal to the sum of the areas of the squares on the other two sides. "
                "This is written mathematically as:\n\n"
                "\\(a^2 + b^2 = c^2\\)\n\n"
                "where \\(c\\) represents the length of the hypotenuse and \\(a\\) and \\(b\\) represent the lengths of the triangle's other two sides. "
                "It is widely used in trigonometry, geometry, and real-world construction calculations."
            ),
            "recommendations": ["What is a derivative?", "How do you solve linear equations?"]
        },
        {
            "query": "What is a derivative?",
            "aliases": ["derivative", "explain derivative", "what does derivative mean", "explain derivatives in calculus"],
            "response": (
                "In calculus, a derivative represents the instantaneous rate of change of a function with respect to one of its variables. "
                "Geometrically, the derivative at a point is the slope of the tangent line to the graph of the function at that point. "
                "Mathematically, the derivative of a function \\(f(x)\\) is defined as the limit:\n\n"
                "\\(f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}\\)\n\n"
                "Derivatives are core to understanding acceleration in physics, marginal costs in economics, and optimization in machine learning."
            ),
            "recommendations": ["Explain the Pythagorean theorem", "How do you solve linear equations?"]
        },
        {
            "query": "How do you solve linear equations?",
            "aliases": ["solve linear equations", "linear equations", "solving linear equations", "explain linear equations"],
            "response": (
                "To solve a linear equation, you isolate the variable on one side of the equal sign by performing inverse operations. "
                "For example, let's solve the linear equation:\n\n"
                "\\(3x + 5 = 11\\)\n\n"
                "1. Subtract 5 from both sides:\n"
                "\\(3x = 11 - 5\\)\n"
                "\\(3x = 6\\)\n\n"
                "2. Divide both sides by 3:\n"
                "\\(x = \\frac{6}{3}\\)\n"
                "\\(x = 2\\)\n\n"
                "This isolates the variable and yields the solution."
            ),
            "recommendations": ["Explain the Pythagorean theorem", "What is a derivative?"]
        },
        {
            "query": "Explain the concept of limits",
            "aliases": ["limits in calculus", "what is a limit", "explain limits", "what are limits"],
            "response": (
                "In mathematics, a limit is the value that a function approaches as the input approaches some value. "
                "Limits are essential to calculus and are used to define continuity, derivatives, and integrals. "
                "We write it as:\n\n"
                "\\(\\lim_{x \\to c} f(x) = L\\)\n\n"
                "This states that as \\(x\\) gets arbitrarily close to \\(c\\), \\(f(x)\\) gets arbitrarily close to the value \\(L\\)."
            ),
            "recommendations": ["What is a derivative?", "Explain integration"]
        },
        {
            "query": "Explain integration",
            "aliases": ["integration", "what is integration", "explain integrals", "integral calculus"],
            "response": (
                "Integration is the reverse process of differentiation. While differentiation finds the rate of change, "
                "integration finds the accumulation of quantities, most commonly the area under a curve. "
                "An indefinite integral is written as:\n\n"
                "\\(\\int f(x) \\, dx = F(x) + C\\)\n\n"
                "where \\(F'(x) = f(x)\\) and \\(C\\) is the constant of integration. "
                "Definite integrals compute the exact net area between two bounds \\(a\\) and \\(b\\)."
            ),
            "recommendations": ["What is a derivative?", "Explain the concept of limits"]
        },
        {
            "query": "What are prime numbers?",
            "aliases": ["prime numbers", "prime number", "explain prime numbers", "what is a prime number"],
            "response": (
                "A prime number is a whole number greater than 1 whose only divisors are 1 and itself. "
                "For example, 2, 3, 5, 7, 11, and 13 are prime numbers because they cannot be factored into other integers. "
                "The number 2 is unique as it is the only even prime number. "
                "Numbers that have more divisors than 1 and themselves are called composite numbers."
            ),
            "recommendations": ["Explain the concept of infinity", "Why is math important?"]
        },
        {
            "query": "Explain the concept of infinity",
            "aliases": ["infinity", "what is infinity", "explain infinity", "is infinity a number"],
            "response": (
                "Infinity is a concept describing something without limit, boundary, or end. "
                "In mathematics, infinity is not a real number but rather a conceptual size. "
                "For example, the set of natural numbers \\(\\{1, 2, 3, ...\\}\\) is infinitely large. "
                "In calculus, we evaluate behavior of functions as variables grow infinitely large: \\(x \\to \\infty\\)."
            ),
            "recommendations": ["What are prime numbers?", "Why is math important?"]
        },
        {
            "query": "Why is math important?",
            "aliases": ["importance of math", "why study math", "how is math useful", "what is math used for"],
            "response": (
                "Mathematics is the foundation of science, engineering, technology, and economics. "
                "It teaches logical reasoning, problem-solving, and abstract thinking. "
                "Without math, modern technologies like computer graphics, internet protocols, machine learning algorithms, "
                "and financial forecasting models would be impossible to construct. It is the language of the universe."
            ),
            "recommendations": ["What are prime numbers?", "Explain the concept of infinity"]
        }
    ]

    for item in math_concepts:
        full_output = (
            f"{item['response']}\n\n"
            "Would you like to ask anything else? Here are other prompts you might like to try:\n"
            + "\n".join([f"- '{rec}'" for rec in item["recommendations"]])
        )
        all_queries = [item["query"]] + item["aliases"]
        for q in all_queries:
            # Normal
            instructions.append({
                "instruction": q,
                "input": "",
                "output": full_output
            })
            # Prepend Persona
            instructions.append({
                "instruction": f"Math Tutor: {q}",
                "input": "",
                "output": f"As a Math Tutor, I would be glad to explain. {full_output}"
            })

    # -------------------------------------------------------------
    # 3. PHYSICS HELPER EXPLANATIONS (20+ samples)
    # -------------------------------------------------------------
    physics_concepts = [
        {
            "query": "What is Newton's second law?",
            "aliases": ["newton's second law", "newton second law", "explain newton's second law", "force equals mass times acceleration"],
            "response": (
                "Newton's second law of motion describes how the velocity of an object changes when it is subjected to an external force. "
                "It states that the acceleration of an object is directly proportional to the net force acting on it "
                "and inversely proportional to its mass. The mathematical equation is:\n\n"
                "\\(F = ma\\)\n\n"
                "where \\(F\\) is the net force vector in Newtons (N), \\(m\\) is the object's mass in kilograms (kg), "
                "and \\(a\\) is the acceleration vector in meters per second squared (m/s²)."
            ),
            "recommendations": ["Explain gravitational potential energy", "How does speed relate to velocity?"]
        },
        {
            "query": "Explain gravitational potential energy",
            "aliases": ["gravitational potential energy", "potential energy", "explain potential energy", "what is gravitational potential energy"],
            "response": (
                "Gravitational potential energy is the energy stored in an object due to its vertical position relative to a reference height. "
                "It represents the work done against gravity to lift the object. The formula is:\n\n"
                "\\(U_g = mgh\\)\n\n"
                "where \\(m\\) is the mass in kg, \\(g\\) is the acceleration due to gravity (approximately \\(9.8 \\text{ m/s}^2\\) on Earth), "
                "and \\(h\\) is the height in meters above the reference point."
            ),
            "recommendations": ["What is Newton's second law?", "How does speed relate to velocity?"]
        },
        {
            "query": "How does speed relate to velocity?",
            "aliases": ["speed vs velocity", "speed and velocity", "difference between speed and velocity", "explain speed and velocity"],
            "response": (
                "In physics, speed and velocity describe motion but have distinct mathematical definitions:\n"
                "- **Speed** is a scalar quantity indicating how fast an object moves, ignoring direction. (e.g., 60 km/h).\n"
                "- **Velocity** is a vector quantity containing both speed and direction of travel (e.g., 60 km/h North).\n\n"
                "Mathematically, velocity is the derivative of position vector with respect to time: \\(\\vec{v} = \\frac{d\\vec{s}}{dt}\\)."
            ),
            "recommendations": ["What is Newton's second law?", "Explain gravitational potential energy"]
        },
        {
            "query": "What is kinetic energy?",
            "aliases": ["kinetic energy", "explain kinetic energy", "formula for kinetic energy", "what is the kinetic energy formula"],
            "response": (
                "Kinetic energy is the energy that an object possesses due to its motion. "
                "It is defined as the work needed to accelerate a body of a given mass from rest to its stated velocity. "
                "The mathematical formula is:\n\n"
                "\\(K_e = \\frac{1}{2}mv^2\\)\n\n"
                "where \\(m\\) is the mass of the object in kilograms, and \\(v\\) is its velocity in meters per second."
            ),
            "recommendations": ["What is Newton's second law?", "Explain gravitational potential energy"]
        },
        {
            "query": "Explain Ohm's law",
            "aliases": ["ohm's law", "explain ohms law", "ohms law formula", "what is ohm's law"],
            "response": (
                "Ohm's law states that the current flowing through a conductor between two points is directly proportional "
                "to the voltage across the two points. The equation representing this relationship is:\n\n"
                "\\(V = IR\\)\n\n"
                "where \\(V\\) is the voltage drop across the resistor in volts (V), \\(I\\) is the current in amperes (A), "
                "and \\(R\\) is the electrical resistance of the material in ohms (\\(\\Omega\\))."
            ),
            "recommendations": ["What is Newton's second law?", "What is kinetic energy?"]
        }
    ]

    for item in physics_concepts:
        full_output = (
            f"{item['response']}\n\n"
            "Would you like to ask anything else? Here are other prompts you might like to try:\n"
            + "\n".join([f"- '{rec}'" for rec in item["recommendations"]])
        )
        all_queries = [item["query"]] + item["aliases"]
        for q in all_queries:
            # Normal
            instructions.append({
                "instruction": q,
                "input": "",
                "output": full_output
            })
            # Prepend Persona
            instructions.append({
                "instruction": f"Physics Helper: {q}",
                "input": "",
                "output": f"As a Physics Helper, let's analyze this. {full_output}"
            })

    # -------------------------------------------------------------
    # 4. RAG CONTEXT-FOLLOWING SAMPLES (18 samples)
    # -------------------------------------------------------------
    rag_scenarios = [
        {
            "instruction": "Explain the derivative using the context.",
            "input": (
                "Web Search Context:\n"
                "- www.khanacademy.org: The derivative of a function describes the function's instantaneous rate of change.\n"
                "- www.derivativecalculus.com: It provides the slope of the line tangent to the graph at a given point."
            ),
            "output": (
                "According to the provided search context, a derivative represents the instantaneous rate of change of a function. "
                "Additionally, it computes the slope of the line tangent to the function's graph at that specific point.\n\n"
                "Would you like to ask anything else? Here are other prompts you can try:\n"
                "- 'Explain the Pythagorean theorem'\n"
                "- 'How do you solve linear equations?'"
            )
        },
        {
            "instruction": "What is Newton's second law based on the text?",
            "input": (
                "Web Search Context:\n"
                "- physics.info: Newton's second law states that force is the rate of change of momentum.\n"
                "- study.com: The force equation is mathematically represented as F = ma."
            ),
            "output": (
                "Based on the provided search context, Newton's second law defines force as the rate of change of momentum. "
                "It is mathematically represented using the force formula \\(F = ma\\), where force equals mass times acceleration.\n\n"
                "Would you like to ask anything else? Here are other prompts you can try:\n"
                "- 'Explain gravitational potential energy'\n"
                "- 'How does speed relate to velocity?'"
            )
        },
        {
            "instruction": "Where did the Pythagorean theorem come from based on the web context?",
            "input": (
                "Web Search Context:\n"
                "- history.math: Named after the ancient Greek mathematician Pythagoras.\n"
                "- mathworld: Although named after him, it was known by Babylonian and Indian mathematicians earlier."
            ),
            "output": (
                "According to the search context, the Pythagorean theorem is named after the ancient Greek mathematician Pythagoras. "
                "However, the context notes that it was actually known by Babylonian and Indian mathematicians prior to Pythagoras.\n\n"
                "Would you like to ask anything else? Here are other prompts you can try:\n"
                "- 'Explain the Pythagorean theorem'\n"
                "- 'What are prime numbers?'"
            )
        },
        {
            "instruction": "What is the speed of gravity according to search results?",
            "input": (
                "Web Search Context:\n"
                "- space.com: Gravity propagates at the speed of light.\n"
                "- nasa.gov: In general relativity, gravitational waves travel at speed c."
            ),
            "output": (
                "Based on the provided search results, gravity propagates at the speed of light. "
                "According to Einstein's theory of general relativity, gravitational waves travel at the speed of light, denoted as \\(c\\).\n\n"
                "Would you like to ask anything else? Here are other prompts you can try:\n"
                "- 'Explain gravitational potential energy'\n"
                "- 'What is Newton's second law?'"
            )
        }
    ]

    for item in rag_scenarios:
        # Standard casing
        instructions.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        })
        # Title casing
        instructions.append({
            "instruction": item["instruction"].title(),
            "input": item["input"],
            "output": item["output"]
        })
        # Slight variation
        instructions.append({
            "instruction": "Use the context to explain: " + item["instruction"],
            "input": item["input"],
            "output": item["output"]
        })

    # Write output to custom_instructions.json
    output_path = os.path.join("data", "custom_instructions.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(instructions, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(instructions)} instruction pairs inside {output_path}!")

if __name__ == "__main__":
    main()

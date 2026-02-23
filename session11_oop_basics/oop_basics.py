"""
Session 11 – Object-Oriented Programming (OOP) Basics in Python
Topics:
  1. Classes and objects
  2. __init__ and self
  3. Instance attributes and instance methods
  4. Class attributes (shared across all instances)
  5. __str__ dunder method
  6. Inheritance and super()
  7. Polymorphism
  8. Abstraction (abc module)
  9. Encapsulation (_private convention)
  10. Bridge preview: OOP applied to data/model handling (leads into Session 12)

Run with: python oop_basics.py
"""

# =============================================================================
# 1. CLASSES AND OBJECTS
# =============================================================================
# A class is a blueprint. An object is an instance created from that blueprint.
#
#   class ClassName:
#       ...
#
# Example: every student shares the same structure (name, age, grade),
# but each student object has its own values.

print("=" * 60)
print("1. Classes and Objects")
print("=" * 60)

class Student:
    def __init__(self, name, age, grade):
        # 'self' refers to the specific instance being created
        self.name  = name   # instance attribute
        self.age   = age
        self.grade = grade

    def greet(self):
        print(f"Hi, I'm {self.name}, age {self.age}, grade {self.grade}.")

s1 = Student("Alice", 20, "A")
s2 = Student("Bob",   22, "B")

s1.greet()   # Hi, I'm Alice, age 20, grade A.
s2.greet()   # Hi, I'm Bob,   age 22, grade B.

# Each object has independent attribute values
print(f"s1.name = {s1.name},  s2.name = {s2.name}")


# =============================================================================
# 2. INSTANCE ATTRIBUTES vs CLASS ATTRIBUTES
# =============================================================================
# - Instance attributes: defined inside __init__ with self.x — unique per object
# - Class attributes:    defined directly in the class body — shared by all instances

print("\n" + "=" * 60)
print("2. Instance vs Class Attributes")
print("=" * 60)

class BankAccount:
    bank_name = "Python Bank"   # class attribute — same for every account

    def __init__(self, owner, balance=0):
        self.owner   = owner    # instance attribute
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"{self.owner} deposited {amount}. New balance: {self.balance}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds.")
        else:
            self.balance -= amount
            print(f"{self.owner} withdrew {amount}. New balance: {self.balance}")

acc1 = BankAccount("Alice", 1000)
acc2 = BankAccount("Bob",    500)

acc1.deposit(200)
acc2.withdraw(100)

print(f"Bank: {BankAccount.bank_name}")   # accessed from the class itself
print(f"Bank: {acc1.bank_name}")          # also accessible from an instance


# =============================================================================
# 3. __str__ DUNDER METHOD
# =============================================================================
# __str__ controls what print(obj) shows.
# Without it, print(obj) shows something like <__main__.BankAccount object at 0x...>

print("\n" + "=" * 60)
print("3. __str__ Dunder Method")
print("=" * 60)

class Product:
    def __init__(self, name, price):
        self.name  = name
        self.price = price

    def __str__(self):
        return f"Product(name='{self.name}', price={self.price})"

p = Product("Laptop", 1200)
print(p)   # Product(name='Laptop', price=1200)


# =============================================================================
# 4. INHERITANCE
# =============================================================================
# A child class inherits all attributes and methods from its parent.
# super().__init__() calls the parent's __init__ to reuse its setup.

print("\n" + "=" * 60)
print("4. Inheritance and super()")
print("=" * 60)

class Animal:
    def __init__(self, name, sound):
        self.name  = name
        self.sound = sound

    def speak(self):
        print(f"{self.name} says: {self.sound}!")

class Dog(Animal):                  # Dog inherits from Animal
    def __init__(self, name):
        super().__init__(name, "Woof")   # reuse Animal's __init__
        self.tricks = []

    def learn_trick(self, trick):
        self.tricks.append(trick)
        print(f"{self.name} learned: {trick}")

    def show_tricks(self):
        if self.tricks:
            print(f"{self.name}'s tricks: {', '.join(self.tricks)}")
        else:
            print(f"{self.name} knows no tricks yet.")

class Cat(Animal):
    def __init__(self, name):
        super().__init__(name, "Meow")

    def speak(self):          # override the parent method
        print(f"{self.name} says: {self.sound}~ (softly)")

dog = Dog("Rex")
cat = Cat("Whiskers")

dog.speak()
cat.speak()                   # overridden version
dog.learn_trick("sit")
dog.learn_trick("shake")
dog.show_tricks()


# =============================================================================
# 5. POLYMORPHISM
# =============================================================================
# Polymorphism = "many forms". The same method name behaves differently
# depending on the object it is called on.
#
# Two flavours in Python:
#   a) Method overriding — child class redefines a parent method (seen in Section 4)
#   b) Duck typing       — if an object has the right method, it just works,
#                          regardless of its class ("if it quacks like a duck…")

print("\n" + "=" * 60)
print("5. Polymorphism")
print("=" * 60)

# --- a) Method overriding ---
# Dog.speak() and Cat.speak() both exist, but do different things.
# We can call speak() on ANY Animal without caring which subclass it is.

animals = [Dog("Rex"), Cat("Whiskers"), Dog("Buddy")]

print("-- method overriding (loop over mixed types) --")
for animal in animals:
    animal.speak()          # correct version called automatically per type

# --- b) Duck typing ---
# This class has NO inheritance from Animal, yet it also has speak().
class Parrot:
    def __init__(self, name, phrase):
        self.name   = name
        self.phrase = phrase

    def speak(self):
        print(f"{self.name} says: {self.phrase}!")

print("\n-- duck typing (no shared base class required) --")
mixed = [Dog("Rex"), Cat("Whiskers"), Parrot("Polly", "Pieces of eight")]
for creature in mixed:
    creature.speak()        # works for all three — they all have speak()


# =============================================================================
# 6. ABSTRACTION
# =============================================================================
# Abstraction hides complex implementation behind a simple interface.
# In Python, the `abc` module lets you define an Abstract Base Class (ABC):
#   - mark methods with @abstractmethod to force subclasses to implement them
#   - trying to instantiate the ABC directly raises a TypeError
#
# Key idea: the ABC says WHAT a class must do; subclasses decide HOW.

print("\n" + "=" * 60)
print("6. Abstraction (Abstract Base Classes)")
print("=" * 60)

from abc import ABC, abstractmethod
import math

class Shape(ABC):                   # inherit from ABC to become abstract
    @abstractmethod
    def area(self):
        """Every shape must implement area()."""
        pass

    @abstractmethod
    def perimeter(self):
        """Every shape must implement perimeter()."""
        pass

    def describe(self):             # concrete method — shared by all shapes
        print(f"{type(self).__name__}: area={self.area():.2f}, "
              f"perimeter={self.perimeter():.2f}")

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width  = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

shapes = [Circle(5), Rectangle(4, 6)]
for s in shapes:
    s.describe()

# Trying to instantiate the abstract class itself raises TypeError
try:
    s = Shape()
except TypeError as e:
    print(f"\nCannot instantiate ABC directly: {e}")


# =============================================================================
# 7. ENCAPSULATION  (_private convention)
# =============================================================================
# Python uses a naming convention:
#   _attribute   →  "protected" — internal use, but accessible
#   __attribute  →  "private"   — name-mangled, harder to access from outside
#
# This signals to other developers: "don't touch this directly."

print("\n" + "=" * 60)
print("7. Encapsulation")
print("=" * 60)

class Thermometer:
    def __init__(self, celsius):
        self._celsius = celsius     # _protected: internal, but readable

    def get_celsius(self):
        return self._celsius

    def set_celsius(self, value):
        if value < -273.15:
            print("Error: below absolute zero!")
        else:
            self._celsius = value

    def get_fahrenheit(self):
        return self._celsius * 9/5 + 32

thermo = Thermometer(25)
print(f"Celsius:    {thermo.get_celsius()} °C")
print(f"Fahrenheit: {thermo.get_fahrenheit()} °F")

thermo.set_celsius(-300)    # Error: below absolute zero!
thermo.set_celsius(37)
print(f"Body temp:  {thermo.get_celsius()} °C")


# =============================================================================
# 8. BRIDGE PREVIEW – OOP FOR DATA & MODELS  (leads into Session 12)
# =============================================================================
# The same OOP ideas apply to ML workflows.
# Instead of passing DataFrames and models around as loose variables,
# we wrap them in classes — just like Session 12's DataHandler / ModelHandler.

print("\n" + "=" * 60)
print("8. Bridge Preview: OOP for ML (preview of Session 12 pattern)")
print("=" * 60)

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class SimpleDataHandler:
    """Wraps raw data loading and train/test splitting."""

    def __init__(self):
        self.X       = None
        self.y       = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load(self):
        iris      = load_iris()
        self.X    = iris.data
        self.y    = iris.target
        print(f"Data loaded: {self.X.shape[0]} rows, {self.X.shape[1]} features")

    def split(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"Train: {len(self.X_train)} rows  |  Test: {len(self.X_test)} rows")


class SimpleModelHandler:
    """Wraps model creation, training, and evaluation."""

    def __init__(self, data_handler: SimpleDataHandler):
        self._data  = data_handler
        self._model = RandomForestClassifier(max_depth=4, random_state=42)

    def train(self):
        self._model.fit(self._data.X_train, self._data.y_train)
        print("Model trained.")

    def evaluate(self):
        preds    = self._model.predict(self._data.X_test)
        accuracy = accuracy_score(self._data.y_test, preds)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy


# --- Run the mini pipeline ---
data  = SimpleDataHandler()
data.load()
data.split()

model = SimpleModelHandler(data)
model.train()
model.evaluate()

print("\nSession 11 completed! → See Session 12 for the full OOP ML pipeline.")

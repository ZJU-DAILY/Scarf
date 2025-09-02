def test_inheritance():
    class Base:
        def __init__(self):
            self.value = 42

        def get_value(self):
            return self.value

    class Derived(Base):
        def __init__(self):
            super().__init__()
            self.value = 84

    derived = Derived()
    print(derived.get_value())  # Should print 84


if __name__ == "__main__":
    test_inheritance()

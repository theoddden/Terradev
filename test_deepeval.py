import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_llm():
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris",
        expected_output="Paris"
    )
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.5)])

import asyncio
import os
import json
from src.agent import FraudCaseDB, FraudAgent

async def test_fraud_agent():
    print("Starting verification...")
    
    # 1. Setup DB
    db_path = os.path.join(os.getcwd(), "src", "fraud_db.json")
    print(f"DB Path: {db_path}")
    
    # Reset DB for testing
    initial_data = [
      {
        "userName": "John",
        "securityIdentifier": "12345",
        "cardEnding": "4242",
        "transactionName": "ABC Industry",
        "transactionAmount": "$125.50",
        "transactionTime": "2:30 PM",
        "transactionLocation": "New York, NY",
        "transactionSource": "alibaba.com",
        "securityQuestion": "What is your mother's maiden name?",
        "securityAnswer": "Smith",
        "status": "pending_review",
        "outcome_note": ""
      }
    ]
    with open(db_path, "w") as f:
        json.dump(initial_data, f, indent=2)

    db = FraudCaseDB(db_path)
    agent = FraudAgent(db)
    
    # 2. Test Lookup
    print("\nTesting Lookup...")
    res = await agent.lookup_user(None, "John")
    print(f"Lookup Result: {res}")
    assert "found a case" in res
    assert agent.current_case["userName"] == "John"

    # 3. Test Verification (Wrong Answer)
    print("\nTesting Verification (Wrong)...")
    res = await agent.verify_security_answer(None, "Jones")
    print(f"Verify Result: {res}")
    assert "incorrect" in res
    assert not agent.verified

    # 4. Test Verification (Correct Answer)
    print("\nTesting Verification (Correct)...")
    res = await agent.verify_security_answer(None, "Smith")
    print(f"Verify Result: {res}")
    assert "Identity verified" in res
    assert agent.verified

    # 5. Test Transaction Confirmation (Safe)
    print("\nTesting Transaction Confirmation (Safe)...")
    res = await agent.process_transaction_response(None, True)
    print(f"Process Result: {res}")
    assert "Marked as safe" in res
    
    # Verify DB update
    with open(db_path, "r") as f:
        data = json.load(f)
        print(f"DB Status: {data[0]['status']}")
        assert data[0]["status"] == "confirmed_safe"

    print("\nVerification Passed!")

if __name__ == "__main__":
    asyncio.run(test_fraud_agent())

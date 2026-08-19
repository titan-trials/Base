"""
Diagnostic: why is Yordan Alvarez not being found?
"""
from pybaseball import playerid_lookup

print("=== Search by last name only: 'Alvarez' ===")
result = playerid_lookup("Alvarez")
print(f"Rows returned: {len(result)}")
if not result.empty:
    print(result[["name_first", "name_last", "key_mlbam", "mlb_played_last"]].to_string())

print("\n=== Search by last name only: 'alvarez' (lowercase) ===")
result2 = playerid_lookup("alvarez")
print(f"Rows returned: {len(result2)}")

print("\n=== Search with fuzzy=True ===")
try:
    result3 = playerid_lookup("alvarez", "yordan", fuzzy=True)
    print(f"Rows returned: {len(result3)}")
    if not result3.empty:
        print(result3[["name_first", "name_last", "key_mlbam"]].to_string())
except TypeError as e:
    print(f"fuzzy param not supported in this pybaseball version: {e}")
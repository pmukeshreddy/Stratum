def first_non_repeating_character(str1: str):
	counts = {}
	for character in str1:
		counts[character] = counts.get(character, 0) + 1

	for character in str1:
		if counts[character] == 1:
			return character
	return None

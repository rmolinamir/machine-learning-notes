/**
 * Pattern or jump table.
 * @param {string} word
 * @return {number[]}
 */
 function buildPatternTable(word: string): Array<number> {
  const patternTable = [0];

  let prefixIndex = 0;
  let suffixIndex = 1;

  while (suffixIndex < word.length) {
    if (word[prefixIndex] === word[suffixIndex]) {
      patternTable[suffixIndex] = prefixIndex + 1;
      suffixIndex += 1;
      prefixIndex += 1;
    } else if (prefixIndex === 0) {
      patternTable[suffixIndex] = 0;
      suffixIndex += 1;
    } else {
      prefixIndex = patternTable[prefixIndex - 1];
    }
  }

  return patternTable;
}

/**
 * The KMP algorithm is a simple substring search algorithm and therefore its purpose is to search for the existence
 * of a substring within a string. To do this, it uses information based on previous matches and failures, taking
 * advantage of the information that the word to search itself contains, to determine where the next existence
 * could occur, without having to analyze the text more than once.
 * @param {string} text
 * @param {string} word
 * @return {number}
 */
export function knuthMorrisPratt(text: string, word: string): number {
  if (word.length === 0) {
    return 0;
  }

  let textIndex = 0;
  let wordIndex = 0;

  const patternTable = buildPatternTable(word);

  while (textIndex < text.length) {
    if (text[textIndex] === word[wordIndex]) {
      // We've found a match.
      if (wordIndex === word.length - 1) {
        return (textIndex - word.length) + 1;
      }
      wordIndex += 1;
      textIndex += 1;
    } else if (wordIndex > 0) {
      wordIndex = patternTable[wordIndex - 1];
    } else {
      wordIndex = 0;
      textIndex += 1;
    }
  }

  return -1;
}

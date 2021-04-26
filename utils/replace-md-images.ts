// Libraries
import {
  resolve,
} from 'path';
import {
  readFileSync,
  writeFileSync,
} from 'fs';
import { Spinner } from 'cli-spinner';

// Helpers
import { MdImagesJsonCache, getFiles } from './helpers';

//
// PARAMS
//

const CACHE_PATH = resolve(__dirname, '.replace-md-images.cache.json');
const SEARCH_DIRECTORY = resolve(__dirname, '..');

//
// CACHE
//

const mdImagesJsonCache = new MdImagesJsonCache({ cachePath: CACHE_PATH });

//
// IMPLEMENTATION
//

const spinner = new Spinner('Fetching images...');

const errors: string[] = [];

/**
 * Transform local URI string to an equivalent Github-like URI string as
 * if the file was uploaded.
 * @param {string} localUri - Local URI string.
 */
function transformLocalUriToGithubUri(localUri: string): string {

  return localUri;
}

/**
 * Replaces all images in a Markdown file with the ones from the cache
 * AFTER transforming them to a Github-like URI as if they were uploaded.
 */
function replaceAndTransformMdImage(file: string, saveDirectory: string): void {
  const directoryCache = mdImagesJsonCache.cache[saveDirectory] || {};

  const cachedImages = Object.keys(directoryCache);

  if (cachedImages.length) {
    const fileContents = readFileSync(file, { encoding: 'utf-8' });

    cachedImages.forEach(i => {
      const src = directoryCache[i];

      const transformedSrc = transformLocalUriToGithubUri(src);

      fileContents.replace(i, transformedSrc);
    });

    writeFileSync(file, fileContents, { flag: 'w+' });
  }
}

/**
 * Replaces all images (from the downloaded cache of images) found in all
 * markdown files within the search directory.
 * @param {string} searchDirectory - Search directory.
 */
async function replaceMarkdownImages(searchDirectory: string = __dirname) {
  spinner.start();

  let index = 0;

  for await (const file of getFiles(searchDirectory)) {
    index ++;
    const parsedFile = file.split('.');
    const fileFormat = parsedFile[parsedFile.length - 1];

    if (fileFormat === 'md') {
      spinner.setSpinnerString(index);

      const directory = file.split('\\');

      const saveDirectory = directory
        .slice(0, directory.length - 1)
        .join('\\');

      replaceAndTransformMdImage(file, saveDirectory);
    }
  }

  spinner.setSpinnerString('slow');

  spinner.stop();
}

// TODO: Parse process.argv to parameterize the search directories.
replaceMarkdownImages(SEARCH_DIRECTORY).then(() => {
  if (errors.length) {
    console.log('\n');
    console.warn('We found the following errors: ', errors);
    console.log('\n');
  }

  console.log('\nDone! Exiting Node.js...');
});

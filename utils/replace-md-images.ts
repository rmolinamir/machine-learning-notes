// Libraries
import {
  resolve,
} from 'path';
import {
  readFileSync,
  writeFileSync,
} from 'fs';
import { Spinner } from 'cli-spinner';
import urljoin from 'url-join';

// Helpers
import { MdImagesJsonCache, getFiles } from './helpers';

//
// PARAMS
//

const CACHE_PATH = resolve(__dirname, '.download-md-images.cache.json');
const SEARCH_DIRECTORY = resolve(__dirname, '..');
const BASE_GITHUB_PATH = 'https://raw.githubusercontent.com/rmolinamir/machine-learning-notes/main';

//
// CACHE
//

const mdImagesJsonCache = new MdImagesJsonCache({ cachePath: CACHE_PATH });

//
// IMPLEMENTATION
//

const spinner = new Spinner('Replacing and transforming images...');

const errors: string[] = [];

/**
 * Transform local URI string to an equivalent Github-like URI string as
 * if the file was uploaded.
 * @param {string} localUri - Local URI string.
 */
function transformLocalUriToGithubUri(localUri: string): string {
  // For example:

  // Transform: C:\\Users\\rober\\Desktop\\Projects\\machine-learning-notes\\images\\Linear-Regression Cheat Sheet.png
  // To: https://github.com/rmolinamir/machine-learning-notes/blob/main/images/Linear-Regression%20Cheat%20Sheet.png

  let uri = localUri;

  uri = uri.replace(SEARCH_DIRECTORY, '');

  uri = uri.replace(/\\|\\\\/g, '/');

  uri = urljoin(BASE_GITHUB_PATH, uri);

  uri = encodeURI(uri);

  return uri;
}

/**
 * Replaces all images in a Markdown file with the ones from the cache
 * AFTER transforming them to a Github-like URI as if they were uploaded.
 */
function replaceAndTransformMdImage(file: string, saveDirectory: string): void {
  const directoryCache = mdImagesJsonCache.cache[saveDirectory] || {};

  const cachedImages = Object.keys(directoryCache);

  if (cachedImages.length) {
    let fileContents = readFileSync(file, { encoding: 'utf-8' });

    cachedImages.forEach(alt => {
      const { local } = directoryCache[alt];

      const transformedSrc = transformLocalUriToGithubUri(local);

      const dynamicRegExp = `!(\\[${alt}\\])\\([^\\s]+\\)`;

      const markdownImageLinkRegExp = new RegExp(dynamicRegExp, 'g');

      fileContents = fileContents.replace(markdownImageLinkRegExp, `![${alt}](${transformedSrc})`);
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

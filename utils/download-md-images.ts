// Libraries
import {
  resolve,
  join,
} from 'path';
import {
  createWriteStream,
  readFileSync,
  mkdirSync,
  existsSync,
} from 'fs';
import marked from 'marked';
import axios from 'axios';
import { JSDOM } from 'jsdom';
import { Spinner } from 'cli-spinner';

// Helpers
import { MdImagesJsonCache, getFiles } from './helpers';

//
// TYPES
//

interface Image {
  alt: string
  title: string
  src: string
}

//
// PARAMS
//

const CACHE_PATH = resolve(__dirname, '.download-md-images.cache.json');
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
 * Parses the Markdown file into a HTML file, then creates a DOM object,
 * then uses querySelectorAll to find all of the images.
 * The images are processed to get their metadata, then we return an image
 * array.
 * @param {string} file - Markdown File.
 */
function findImagesInMarkdown(file: string): Array<Image> {
  const readMe = readFileSync(file, { encoding: 'utf-8' });

  const readmeHtml = marked(readMe); // .md to .html.

  const dom = new JSDOM(readmeHtml);// .html to DOM.

  const images: Array<Image> = Array // Parsing images in the DOM.
    .prototype
    .slice
    .call(dom.window.document.querySelectorAll('img'))
    .reduce((rawImages: Array<Image>, image: any) => {
      rawImages.push({
        alt: image.getAttribute('alt') as string,
        title: image.getAttribute('title') as string,
        src: image.getAttribute('src') as string,
      });
      return rawImages;
    }, []);

  return images;
}

/**
 * Downloads all of the found images in the markdown files,
 * then save the data in the passed directory.
 * @param {Image[]} images - Array of images attrs.
 * @param {string} saveDirectory - Download directory.
 */
async function downloadImages(images: Image[], saveDirectory: string) {
  for (let i = 0; i < images.length; i++) {
    const { src, title, alt } = images[i];

    const filename = `${title ? `${title}_${alt}` : alt}.png`
      .replace(/[/\\?%*:|"<>]/g, '')
      .replace(' ', '-');

    const cachedImage = mdImagesJsonCache.get(saveDirectory, alt);

    try {
      // Fetch the image ONLY if it hasn't been cached, or if the cached source
      // has changed.
      if (!cachedImage || cachedImage.src !== src) {
        const result = await axios({ url: src, responseType: 'stream' });

        // Downloading the images, then saving them.
        await new Promise(resolve => {
          // Check if path exists first, if not then create the images dir.
          if (!existsSync(join(saveDirectory, 'images'))) {
            mkdirSync(join(saveDirectory, 'images'));
          }

          const filePath = join(saveDirectory, 'images', filename);
  
          result.data
            .pipe(createWriteStream(
              filePath,
              { flags: 'w+' },
            ))
            .on('finish', (value: unknown) => {
              mdImagesJsonCache.set(saveDirectory, alt, { src, local: filePath });
  
              resolve(value);
            })
            .on('error', (err: Error) => {
              // throw new Error(_.message);
              errors.push(`Fetching image [${src}] resulted in the following error: ${err.message}`);
            });
        });
      }
    } catch (err) {
      errors.push(`Fetching image [${src}] resulted in the following error: ${err.message}`);
    } finally {
      // Save the cache after each iteration.
      mdImagesJsonCache.write();
    }
  }
}

/**
 * Downloads all images then saves them in an images directory
 * at the same scope of the found markdown file.
 * @param {string} searchDirectory - Search directory.
 */
async function downloadMarkdownImages(searchDirectory: string = __dirname) {
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

      const images = findImagesInMarkdown(file);

      await downloadImages(images, saveDirectory);
    }
  }

  spinner.setSpinnerString('slow');

  spinner.stop();
}

// TODO: Parse process.argv to parameterize the search directories.
downloadMarkdownImages(SEARCH_DIRECTORY).then(() => {
  if (errors.length) {
    console.log('\n');
    console.warn('We found the following errors: ', errors);
    console.log('\n');
  }

  console.log('\nDone, exiting Node.js.');
});

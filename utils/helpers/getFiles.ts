// Libraries
import { resolve } from 'path';
import { promises } from 'fs';

const { readdir } = promises;

const DIR_BLACKLIST = [
  'node_modules',
  'jupyter-notebooks',
  '.git',
];

/**
 * Recursive Files Async Iterator Generator. Allow us to iterate over data that
 * comes asynchronously, in this case the data will be our directories.
 * Returns an Async Iterator of the files in every directory that
 * is not a node module or a git folder.
 * @param {string} dir - Directory.
 */
export async function* getFiles(dir: string): AsyncGenerator<string> {
  // A representation of a directory entry, as
  // returned by reading from an fs.Dir.
  const dirents = await readdir(dir, { withFileTypes: true });

  for (const dirent of dirents) { // Base condition.
    // Resolving the files and directories.
    const res = resolve(dir, dirent.name);

    // // Do not get files if they're node modules or git folders.
    // const shouldNotGetFiles = (
    //   res.includes('node_modules') ||
    //   res.includes('node_modules') ||
    //   res.includes('.git')
    // );

    // Do not get files if they're node modules or git folders.
    const shouldNotGetFiles = DIR_BLACKLIST.reduce(
      (bool, dir) => bool || res.includes(dir),
      false,
    );

    // If it's a directory, recursively get more files.
    if (dirent.isDirectory() && !shouldNotGetFiles) {
      yield* getFiles(res);
    } else {
      yield res; // Return the file.
    }
  }
}

// Libraries
import {
  readFileSync,
  writeFileSync,
} from 'fs';

interface Cache<T> {
  cache: Record<string, T>;

  get(...args: unknown[]): string;

  exists(...args: unknown[]): boolean;

  set(...args: unknown[]): void;

  write(...args: unknown[]): void;
}

export class MdImagesJsonCache implements Cache<Record<string, string>> {
  private cachePath: string;

  public cache: Record<string, Record<string, string>> = {};

  constructor(args: {
    cachePath: string;
  }) {
    this.cachePath = args.cachePath;

    this.cache = JSON.parse(readFileSync(
      this.cachePath,
      { encoding: 'utf-8' },
    )) as Record<string, Record<string, string>>;
  }

  public get(saveDirectory: string, src: string): string {
    return this.cache[saveDirectory] && this.cache[saveDirectory][src];
  }

  public exists(saveDirectory: string, src: string): boolean {
    return Boolean(this.get(saveDirectory, src));
  }

  public set(saveDirectory: string, src: string, filePath: string): void {
    if (!this.cache[saveDirectory]) {
      this.cache[saveDirectory] = {};
    }

    this.cache[saveDirectory][src] = filePath;
  }

  public write(): void {
    writeFileSync(this.cachePath, JSON.stringify(this.cache, null, 2));
  }
}

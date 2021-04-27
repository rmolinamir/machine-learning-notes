// Libraries
import {
  readFileSync,
  writeFileSync,
} from 'fs';

interface Cache<T> {
  cache: Record<string, T>;

  get(...args: unknown[]): unknown;

  exists(...args: unknown[]): boolean;

  set(...args: unknown[]): void;

  write(...args: unknown[]): void;
}

export class MdImagesJsonCache implements Cache<Record<string, { src: string; local: string; }>> {
  private cachePath: string;

  public cache: Record<string, Record<string, { src: string; local: string; }>> = {};

  constructor(args: {
    cachePath: string;
  }) {
    this.cachePath = args.cachePath;

    this.cache = JSON.parse(readFileSync(
      this.cachePath,
      { encoding: 'utf-8' },
    ));
  }

  public get(saveDirectory: string, alt: string): { src: string; local: string; } {
    return this.cache[saveDirectory] && this.cache[saveDirectory][alt];
  }

  public exists(saveDirectory: string, src: string): boolean {
    return Boolean(this.get(saveDirectory, src));
  }

  public set(saveDirectory: string, alt: string, val: { src: string; local: string; }): void {
    if (!this.cache[saveDirectory]) {
      this.cache[saveDirectory] = {};
    }

    this.cache[saveDirectory][alt] = val;
  }

  public write(): void {
    writeFileSync(this.cachePath, JSON.stringify(this.cache, null, 2));
  }
}

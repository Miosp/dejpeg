// @jsquash/* are lazy dynamic imports; they are not installed dependencies
// yet (see spec: Task 4 adds them as optional peer deps). These bare module
// declarations let typecheck pass; the imports never resolve in test env
// because the native decode path always succeeds first.
declare module "@jsquash/heic";
declare module "@jsquash/tiff";
declare module "@jsquash/avif";
